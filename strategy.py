#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3, close > 1d EMA34, and volume > 2.0x 20-bar avg.
# Short when price breaks below S3, close < 1d EMA34, and volume > 2.0x 20-bar avg.
# Exit when price re-enters the Camarilla range (between S3 and R3).
# Uses 12h timeframe for optimal trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Camarilla levels provide high-probability reversal/breakout points from 1d OHLC.
# 1d EMA34 filters for higher timeframe trend alignment.
# Volume confirmation with higher threshold reduces false breakouts.
# Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Need to align 1d OHLC to 12h bars: use previous completed 1d bar's OHLC
    prev_close_1d = df_1d['close'].shift(1).values  # previous 1d close
    prev_high_1d = df_1d['high'].shift(1).values    # previous 1d high
    prev_low_1d = df_1d['low'].shift(1).values      # previous 1d low
    prev_open_1d = df_1d['open'].shift(1).values    # previous 1d open
    
    # Typical price for Camarilla calculation
    typical_price = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_hl = prev_high_1d - prev_low_1d
    
    # Camarilla levels
    R3 = close_1d + range_hl * 1.1 / 4
    S3 = close_1d - range_hl * 1.1 / 4
    R4 = close_1d + range_hl * 1.1 / 2
    S4 = close_1d - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: volume > 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_R4 = R4_aligned[i]
        curr_S4 = S4_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, close > 1d EMA34, volume spike
            if (curr_close > curr_R3 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, close < 1d EMA34, volume spike
            elif (curr_close < curr_S3 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price re-enters the Camarilla range (below R3)
            if curr_close < curr_R3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price re-enters the Camarilla range (above S3)
            if curr_close > curr_S3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals