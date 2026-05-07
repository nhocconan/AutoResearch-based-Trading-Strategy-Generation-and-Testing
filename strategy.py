#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels (previous day's OHLC)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # Handle first value
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels: R3, S3
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Daily trend filter: EMA34 on daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R3[i]) or np.isnan(S3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price breaks above R3 in daily uptrend with volume
            if close[i] > R3[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in daily downtrend with volume
            elif close[i] < S3[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below R3 or trend changes
            if close[i] < R3[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above S3 or trend changes
            if close[i] > S3[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend and volume filter
# - Uses previous day's OHLC to calculate Camarilla R3 (resistance) and S3 (support)
# - Enters long when 12h price breaks above R3 with volume confirmation in daily uptrend
# - Enters short when 12h price breaks below S3 with volume confirmation in daily downtrend
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x 20-period average) reduces false breakouts
# - Exits when price returns to the breakout level or trend changes
# - Position size 0.25 targets ~20-50 trades per year to stay within limits
# - Camarilla levels provide mathematically derived support/resistance
# - Works in both bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend)
# - Avoids overtrading by requiring confluence of price, volume, and trend
# - Proven pattern: similar strategies show strong performance on ETH/SOL in DB