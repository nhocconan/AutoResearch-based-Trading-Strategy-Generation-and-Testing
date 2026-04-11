# 4h_12h_kama_volatility_breakout_v1
# Strategy: 4h Kaufman Adaptive Moving Average (KAMA) breakout with volume confirmation and 12h volatility filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in ranging markets. Breakouts above/below KAMA with volume confirmation capture trends. 12h ATR-based volatility filter avoids low-volatility chop. Designed for low trade frequency (<30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_kama_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # KAMA(10, 2, 30) calculation
    close_series = pd.Series(close)
    change = np.abs(close_series.diff(10))
    volatility = close_series.diff().abs().rolling(10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [np.nan] * len(close)
    kama[9] = close.iloc[9]  # seed
    for i in range(10, len(close)):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama = np.array(kama)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(atr_14_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: only trade when 12h ATR > its 50-period average (avoid low-vol chop)
        if i < 50:
            vol_filter = True  # warmup period
        else:
            vol_avg_50 = np.nanmean(atr_14_12h_aligned[i-50:i])
            vol_filter = not np.isnan(vol_avg_50) and atr_14_12h_aligned[i] > vol_avg_50
        
        # Entry logic: KAMA breakout + volume + volatility filter
        if close[i] > kama[i] and vol_confirm[i] and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < kama[i] and vol_confirm[i] and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses back through KAMA
        elif position == 1 and close[i] <= kama[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= kama[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals