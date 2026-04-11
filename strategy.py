#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Long: Lips > Teeth > Jaw (bullish alignment) + price > Lips + 1w close > 1w EMA(21) + volume > 1.5x 20-period avg
# - Short: Lips < Teeth < Jaw (bearish alignment) + price < Lips + 1w close < 1w EMA(21) + volume > 1.5x 20-period avg
# - Exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw crossover) or price closes inside Alligator mouth
# - Uses 1w trend filter to avoid counter-trend trades in strong trends
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Alligator catches trends early; 1w filter ensures alignment with higher timeframe trend

name = "12h_1w_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return signals
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute 12h Williams Alligator
    # Median price = (high + low + close) / 3
    median_price = (high + low + close) / 3
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Alligator lines
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1w trend filter: close above/below 1w EMA(21)
        trend_filter_long = close_price > ema_21_1w_aligned[i]
        trend_filter_short = close_price < ema_21_1w_aligned[i]
        
        # Alligator alignment conditions
        # Bullish: Lips > Teeth > Jaw
        bullish_alignment = lips_val > teeth_val > jaw_val
        # Bearish: Lips < Teeth < Jaw
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        # Mouth open/closed detection
        # Mouth closed when Lips crosses Teeth or Jaw, or price inside Alligator
        lips_teeth_cross = (lips_val <= teeth_val and 
                           (i == 100 or lips[i-1] > teeth[i-1])) or \
                          (lips_val >= teeth_val and 
                           (i == 100 or lips[i-1] < teeth[i-1]))
        teeth_jaw_cross = (teeth_val <= jaw_val and 
                          (i == 100 or teeth[i-1] > jaw[i-1])) or \
                         (teeth_val >= jaw_val and 
                          (i == 100 or teeth[i-1] < jaw[i-1]))
        price_inside_mouth = (close_price >= min(lips_val, jaw_val) and 
                             close_price <= max(lips_val, jaw_val))
        
        # Exit conditions: mouth closes or price inside mouth
        exit_long = lips_teeth_cross or teeth_jaw_cross or price_inside_mouth
        exit_short = lips_teeth_cross or teeth_jaw_cross or price_inside_mouth
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long entry: bullish alignment + price above lips + volume confirmation + long trend filter
        if (bullish_alignment and 
            close_price > lips_val and 
            vol_confirm and 
            trend_filter_long):
            enter_long = True
        
        # Short entry: bearish alignment + price below lips + volume confirmation + short trend filter
        if (bearish_alignment and 
            close_price < lips_val and 
            vol_confirm and 
            trend_filter_short):
            enter_short = True
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals