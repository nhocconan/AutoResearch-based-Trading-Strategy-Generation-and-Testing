#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 12h trend filter and volume confirmation
# Long when price breaks above R4 AND price > 12h EMA50 AND volume > 1.8x 24-bar avg
# Short when price breaks below S4 AND price < 12h EMA50 AND volume > 1.8x 24-bar avg
# Exit when price crosses Camarilla H3/L3 levels (mean reversion toward median)
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining edge.
# Target: 12-30 trades/year on 6h (48-120 total over 4 years).
# Camarilla R4/S4 represent strong breakout levels; 12h EMA50 filters counter-trend moves.
# Volume confirmation ensures institutional participation, reducing false breakouts.

name = "6h_Camarilla_R4S4_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels (prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract daily OHLC values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily OHLC to 6h timeframe (each value represents the prior day's close)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate Camarilla levels for each 6h bar based on prior day's OHLC
    # Camarilla R4/S4 and H3/L3 levels
    daily_range = daily_high_aligned - daily_low_aligned
    camarilla_h3 = daily_close_aligned + daily_range * 1.1 / 4
    camarilla_l3 = daily_close_aligned - daily_range * 1.1 / 4
    camarilla_r4 = daily_close_aligned + daily_range * 1.1 / 2
    camarilla_s4 = daily_close_aligned - daily_range * 1.1 / 2
    
    # Volume confirmation: >1.8x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.8 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # EMA50 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_s4[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_12h_aligned[i]
        
        # Camarilla levels
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below H3 (mean reversion to median)
            if curr_close < h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above L3 (mean reversion to median)
            if curr_close > l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R4 AND price > 12h EMA50 AND volume confirmation
            if curr_close > r4 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S4 AND price < 12h EMA50 AND volume confirmation
            elif curr_close < s4 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals