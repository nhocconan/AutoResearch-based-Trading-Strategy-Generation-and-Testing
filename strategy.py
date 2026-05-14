#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction + RSI(14) mean reversion + 1w volume spike filter.
# Long when KAMA rising (bullish trend) AND RSI < 30 (oversold) AND 1w volume > 1.5x 20-period average.
# Short when KAMA falling (bearish trend) AND RSI > 70 (overbought) AND 1w volume > 1.5x 20-period average.
# Exit on opposite RSI condition (RSI > 50 for longs, RSI < 50 for shorts).
# Uses 1w HTF for volume confirmation to reduce noise and overtrading vs shorter timeframes.
# KAMA adapts to market efficiency, effective in both bull and bear markets.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.

name = "1d_KAMA_RSI_1wVolumeFilter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # 1d KAMA(10) - adaptive trend
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.diff(kama, prepend=kama[0])  # >0 rising, <0 falling
    
    # 1d RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    
    # 1w volume confirmation: > 1.5x 20-period average
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = volume_1w > (1.5 * vol_ma_20_1w)
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(kama_dir[i]) or
            np.isnan(rsi[i]) or
            np.isnan(volume_confirm_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising AND RSI < 30 AND 1w volume confirm
            if (kama_dir[i] > 0 and 
                rsi[i] < 30 and 
                volume_confirm_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling AND RSI > 70 AND 1w volume confirm
            elif (kama_dir[i] < 0 and 
                  rsi[i] > 70 and 
                  volume_confirm_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (mean reversion complete)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (mean reversion complete)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals