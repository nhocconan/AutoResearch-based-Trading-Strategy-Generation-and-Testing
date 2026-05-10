# 12h_1w1d_Trend_With_Volume_Confirmation
# Hypothesis: On 12h timeframe, use 1w EMA50 trend filter and 1d volume confirmation for EMA21 breakout entries.
# The strategy enters long when price breaks above EMA21 with volume > 1.5x 24-period average and 1w uptrend.
# Exits when price breaks below EMA21 or 1w trend turns down. Designed for fewer trades (~20-50/year) to avoid fee drag.

name = "12h_1w1d_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # 1d volume confirmation (1.5x 24-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.zeros_like(volume_1d)
    vol_sum = 0
    for i in range(len(volume_1d)):
        vol_sum += volume_1d[i]
        if i >= 24:
            vol_sum -= volume_1d[i-24]
        if i >= 23:
            vol_ma_1d[i] = vol_sum / 24
        else:
            vol_ma_1d[i] = np.nan
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align 1d volume confirmation to 12h
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # EMA21 on 12h for entry/exit
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or np.isnan(ema21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above EMA21 with volume confirmation and 1w uptrend
            if (close[i] > ema21[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below EMA21 with volume confirmation and 1w downtrend
            elif (close[i] < ema21[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below EMA21 or 1w trend turns down
            if (close[i] < ema21[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above EMA21 or 1w trend turns up
            if (close[i] > ema21[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals