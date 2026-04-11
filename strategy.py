#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Williams Alligator + 1w EMA filter + volume confirmation
# - Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs on median price
# - Long: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA(50) AND volume > 1.5x 20-day avg
# - Short: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA(50) AND volume > 1.5x 20-day avg
# - Exit: When Alligator alignment breaks (Lips crosses Teeth or Jaw)
# - Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# - Works in both bull and bear markets by capturing strong trends with Alligator alignment
# - Volume confirmation ensures breakouts have conviction
# - 1w EMA filter prevents trading against higher timeframe trend

name = "1d_1w_alligator_volume_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute Williams Alligator components
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
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
    
    # Pre-compute volume confirmation (20-day average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup (for Alligator and 1w EMA)
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # 1w EMA trend filter
        ema_bias_long = close_price > ema_50_1w_aligned[i]
        ema_bias_short = close_price < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish Alligator alignment + price above 1w EMA + volume confirmation
        if bullish_alignment and ema_bias_long and vol_confirm:
            enter_long = True
        
        # Short: bearish Alligator alignment + price below 1w EMA + volume confirmation
        if bearish_alignment and ema_bias_short and vol_confirm:
            enter_short = True
        
        # Exit conditions: When Alligator alignment breaks
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bullish alignment breaks
            exit_long = not bullish_alignment
        elif position == -1:
            # Exit short if bearish alignment breaks
            exit_short = not bearish_alignment
        
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