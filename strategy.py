#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA50 trend filter + volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1w EMA50 AND volume > 1.8x 20-bar avg
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1w EMA50 AND volume > 1.8x 20-bar avg
# Exit when Alligator alignment breaks (jaws-teeth-lips not in order)
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 15-30 trades/year on 1d timeframe (60-120 total over 4 years) to avoid overtrading.
# Williams Alligator identifies trend phases via smoothed medians; 1w EMA50 filters for higher-timeframe trend alignment.
# Works in bull markets by capturing sustained uptrends and in bear markets by capturing downtrends with trend alignment preventing counter-trend trades.

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
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
    
    # Williams Alligator: Smoothed medians (Jaw=13, Teeth=8, Lips=5)
    # Jaw: Smoothed median of 13 periods, shifted 8 bars forward
    # Teeth: Smoothed median of 8 periods, shifted 5 bars forward  
    # Lips: Smoothed median of 5 periods, shifted 3 bars forward
    median = (high + low) / 2.0
    
    # Calculate smoothed medians using rolling median with smoothing
    jaw_raw = pd.Series(median).rolling(window=13, min_periods=13).median().values
    teeth_raw = pd.Series(median).rolling(window=8, min_periods=8).median().values
    lips_raw = pd.Series(median).rolling(window=5, min_periods=5).median().values
    
    # Apply forward shifts (Alligator specific)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set NaN for shifted values that rolled from end
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # Alligator jaw and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (not bullish: jaw < teeth < lips)
            if not (curr_jaw < curr_teeth < curr_lips):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (not bearish: jaw > teeth > lips)
            if not (curr_jaw > curr_teeth > curr_lips):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Alligator bullish alignment AND price > 1w EMA50 AND volume confirmation
            if curr_jaw < curr_teeth < curr_lips and curr_close > curr_ema50_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator bearish alignment AND price < 1w EMA50 AND volume confirmation
            elif curr_jaw > curr_teeth > curr_lips and curr_close < curr_ema50_1w and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals