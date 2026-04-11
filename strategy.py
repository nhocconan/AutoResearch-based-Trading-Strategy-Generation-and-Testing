#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with volume confirmation
# - Uses 1w EMA(13,8,5) smoothed (Alligator Jaw/Teeth/Lips) for trend direction
# - Elder Ray: Bull/Bear Power = Close - EMA13 (bullish) / EMA13 - Close (bearish)
# - Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND volume > 1.5x 20-bar avg
# - Short: Jaw > Teeth > Lips (bearish alignment) AND Bear Power > 0 AND volume > 1.5x 20-bar avg
# - Exit: Opposite Alligator alignment OR power crosses zero
# - Position size: ±0.25 discrete to limit drawdown and fee churn
# - Target: 15-30 trades/year (60-120 total over 4 years) - well within fee drag limits
# - Works in both bull/bear: Alligator catches trends, Elder Ray filters false breakouts

name = "12h_1w_alligator_elder_volume_v1"
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
    
    # Load 1w data ONCE before loop for Alligator and Elder Ray (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Pre-compute 1w OHLC
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Alligator: SMAs of median price (HL/2) with different periods
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # All smoothed with 8, 5, 3 bars respectively (but we'll use simple EMA for proxy)
    median_price = (high_1w + low_1w) / 2
    
    # Jaw (blue): 13-period EMA of median, smoothed 8 periods
    jaw_raw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw_raw.ewm(span=8, adjust=False, min_periods=8).mean()
    
    # Teeth (red): 8-period EMA of median, smoothed 5 periods
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth_raw.ewm(span=5, adjust=False, min_periods=5).mean()
    
    # Lips (green): 5-period EMA of median, smoothed 3 periods
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips_raw.ewm(span=3, adjust=False, min_periods=3).mean()
    
    # Elder Ray Power
    # Bull Power = Close - EMA13
    # Bear Power = EMA13 - Close
    ema_13 = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = close_1w - ema_13
    bear_power = ema_13 - close_1w
    
    # Align all 1w indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips.values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power.values)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power.values)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Alligator values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Elder Ray values
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Alligator alignment
        bullish_alignment = lips_val > teeth_val > jaw_val  # Lips > Teeth > Jaw
        bearish_alignment = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        # Elder Ray conditions
        bull_power_positive = bull_power_val > 0
        bear_power_positive = bear_power_val > 0
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish alignment AND bull power positive AND volume confirmation
        if bullish_alignment and bull_power_positive and vol_confirm:
            enter_long = True
        
        # Short: bearish alignment AND bear power positive AND volume confirmation
        if bearish_alignment and bear_power_positive and vol_confirm:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bearish alignment OR bull power turns negative
            exit_long = (not bullish_alignment) or (bull_power_val <= 0)
        elif position == -1:
            # Exit short if bullish alignment OR bear power turns negative
            exit_short = (not bearish_alignment) or (bear_power_val <= 0)
        
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