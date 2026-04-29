#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above R4 AND close > 1d EMA34 AND volume > 1.8x 20-bar avg
# Short when price breaks below S4 AND close < 1d EMA34 AND volume > 1.8x 20-bar avg
# Exit when price retouches the central pivot (P) level
# Uses discrete position sizing (0.25) to reduce fee drag while maintaining profitability.
# Target: 80-150 total trades over 4 years (20-38/year) on 4h.
# R4/S4 levels provide stronger breakout confirmation than R3/S3, reducing false signals.
# 1d EMA34 filters counter-trend moves more effectively than shorter EMAs.
# Volume confirmation ensures institutional participation, reducing whipsaws.
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).

name = "4h_Camarilla_R4S4_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: P = (H + L + C) / 3
    # R4 = C + (H - L) * 1.1
    # S4 = C - (H - L) * 1.1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    p = (high_1d + low_1d + close_1d) / 3.0
    # R4 and S4 levels
    r4 = close_1d + (high_1d - low_1d) * 1.1
    s4 = close_1d - (high_1d - low_1d) * 1.1
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    p_aligned = align_htf_to_ltf(prices, df_1d, p)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA warmup and EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(p_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34 = ema_34_1d_aligned[i]
        curr_p = p_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retouches central pivot P
            if curr_close <= curr_p:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retouches central pivot P
            if curr_close >= curr_p:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R4 AND close > 1d EMA34 AND volume confirmation
            if curr_close > curr_r4 and close[i] > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S4 AND close < 1d EMA34 AND volume confirmation
            elif curr_close < curr_s4 and close[i] < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals