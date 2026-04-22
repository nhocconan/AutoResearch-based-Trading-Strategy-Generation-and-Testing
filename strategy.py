# 1h_Combined_Momentum_Regime_V2
# Hypothesis: Use 1d trend filter (price > EMA50 for long, < EMA50 for short) + 4h momentum (MACD histogram cross) + 1h entry timing with volume confirmation.
# Trend filter from higher timeframe reduces false signals in choppy markets. Momentum captures short-term moves. Volume confirms conviction.
# Designed to work in both bull and bear by only taking trades aligned with daily trend.
# Target trade frequency: 15-30/year per symbol.

#!/usr/bin/env python3
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
    
    # --- Higher Timeframe Data (loaded ONCE before loop) ---
    # 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h for momentum
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 4h Momentum: MACD Histogram (12,26,9) ---
    close_4h = df_4h['close'].values
    ema12 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_4h).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    macd_hist_aligned = align_htf_to_ltf(prices, df_4h, macd_hist)
    
    # --- 1h Filters: Volume Average (20-period) ---
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- Session Filter: 08-20 UTC ---
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup for indicators
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(macd_hist_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Daily uptrend (price > EMA50) + MACD hist crosses above zero + volume spike
            if (close[i] > ema50_1d_aligned[i] and 
                macd_hist_aligned[i] > 0 and 
                macd_hist_aligned[i-1] <= 0 and  # crossed above zero this bar
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: Daily downtrend (price < EMA50) + MACD hist crosses below zero + volume spike
            elif (close[i] < ema50_1d_aligned[i] and 
                  macd_hist_aligned[i] < 0 and 
                  macd_hist_aligned[i-1] >= 0 and  # crossed below zero this bar
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Daily trend turns down OR MACD hist crosses below zero
                if (close[i] <= ema50_1d_aligned[i] or 
                    macd_hist_aligned[i] < 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: Daily trend turns up OR MACD hist crosses above zero
                if (close[i] >= ema50_1d_aligned[i] or 
                    macd_hist_aligned[i] > 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Combined_Momentum_Regime_V2"
timeframe = "1h"
leverage = 1.0