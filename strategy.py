# 1h_RSI_TFS_Trend_Filter
# Hypothesis: RSI(14) with 4h/1d trend filter and volume confirmation. Long when RSI crosses above 40 in uptrend (4h/1d EMA50), short when RSI crosses below 60 in downtrend. Volume confirmation filters low-probability entries. Works in bull by buying pullbacks, in bear by selling rallies. Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
# Uses 4h EMA50 for trend, 1d EMA50 for higher timeframe trend confirmation, and 1d volume spike for entry confirmation.

name = "1h_RSI_TFS_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    rsi_period = 14
    rsi = np.full(n, np.nan)
    if n >= rsi_period:
        delta = np.diff(close)
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = np.full(n, np.nan)
        roll_down = np.full(n, np.nan)
        roll_up[rsi_period-1] = np.mean(up[:rsi_period])
        roll_down[rsi_period-1] = np.mean(down[:rsi_period])
        for i in range(rsi_period, n):
            roll_up[i] = (roll_up[i-1] * (rsi_period-1) + up[i]) / rsi_period
            roll_down[i] = (roll_down[i-1] * (rsi_period-1) + down[i]) / rsi_period
        rs = np.where(roll_down != 0, roll_up / roll_down, 0)
        rsi[rsi_period-1:] = 100 - (100 / (1 + rs))
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema50_4h[i-1]
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA50 for higher timeframe trend confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_period, 50)  # Ensure RSI and EMA50 ready
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 1d volume (scaled)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 24.0  # 24x 1h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend determination: price vs 4h EMA50 and 1d EMA50
        is_uptrend = close[i] > ema50_4h_aligned[i] and close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_4h_aligned[i] and close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: RSI crosses above 40 in uptrend with volume confirmation
            if (i > start_idx and 
                rsi[i-1] <= 40 and rsi[i] > 40 and
                is_uptrend and
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: RSI crosses below 60 in downtrend with volume confirmation
            elif (i > start_idx and 
                  rsi[i-1] >= 60 and rsi[i] < 60 and
                  is_downtrend and
                  volume_confirm):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI crosses below 50 or trend changes to downtrend
            if (i > start_idx and 
                (rsi[i] < 50 or not is_uptrend)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI crosses above 50 or trend changes to uptrend
            if (i > start_idx and 
                (rsi[i] > 50 or not is_downtrend)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals