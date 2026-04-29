#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND price > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when Jaw > Teeth > Lips (bearish alignment) AND price < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exit when Alligator alignment breaks (Jaw-Teeth-Lips not ordered) OR price crosses 1w EMA50
# Uses discrete position sizing (0.25) to reduce fee drag.
# Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years) to avoid overtrading.
# Williams Alligator identifies trend via smoothed medians; works in bull via bullish alignment,
# in bear via bearish alignment with 1w trend filter preventing counter-trend trades.

name = "1d_WilliamsAlligator_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA (smoothed MA)
    # SMMA is similar to EMA but with different smoothing; we'll use EMA as approximation
    # Jaw: 13-period EMA, smoothed 8 bars ahead
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw.ewm(span=8, adjust=False, min_periods=8).mean().values  # additional smoothing
    
    # Teeth: 8-period EMA, smoothed 5 bars ahead
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: 5-period EMA, smoothed 3 bars ahead
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: >2.0x 20-bar average volume (strict to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13+8, 8+5, 5+3)  # volume MA and Alligator warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_close = close[i]
        
        # Check Alligator alignment
        bullish_alignment = curr_jaw < curr_teeth < curr_lips  # Jaw < Teeth < Lips
        bearish_alignment = curr_jaw > curr_teeth > curr_lips  # Jaw > Teeth > Lips
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks OR price crosses below 1w EMA50
            if not bullish_alignment or curr_close < curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks OR price crosses above 1w EMA50
            if not bearish_alignment or curr_close > curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bullish alignment AND price > 1w EMA50 AND volume confirmation
            if bullish_alignment and curr_close > curr_ema50_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bearish alignment AND price < 1w EMA50 AND volume confirmation
            elif bearish_alignment and curr_close < curr_ema50_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals