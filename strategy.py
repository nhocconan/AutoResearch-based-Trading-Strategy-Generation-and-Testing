#!/usr/bin/env python3
"""
12h_1d_kama_trend_v1
Strategy: 12h KAMA direction with 1d RSI and volatility filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 12h for trend direction, filtered by 1d RSI (avoiding extremes) and 1d volatility regime (low ATR ratio). Designed to capture trending moves while avoiding whipsaws in chop and overextended reversals. Works in bull markets by following uptrends and in bear markets by avoiding false longs during downtrends. Target: 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h KAMA(14, 2, 30) for trend direction
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[:10] = np.nan  # First 10 values invalid
    
    volatility = np.abs(np.diff(close, prepend=np.nan))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    
    er = change / volatility_sum
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama = np.where(np.isnan(kama), close, kama)  # Fill initial NaNs
    
    # 1d RSI(14) for overbought/oversold filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan] * 14, rsi[14:]])  # Align with original index
    
    # 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    atr_ratio = np.concatenate([[np.nan] * 50, atr_ratio[50:]])  # Align with original index
    
    # Align HTF indicators to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after KAMA warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters
        kama_up = price_close > kama[i]
        kama_down = price_close < kama[i]
        
        # RSI filter: avoid extremes (>70 or <30)
        rsi_ok = (rsi_aligned[i] >= 30) & (rsi_aligned[i] <= 70)
        
        # Volatility filter: low volatility regime (avoid chop)
        vol_ok = atr_ratio_aligned[i] < 1.5  # ATR below 1.5x its average
        
        # Long: KAMA uptrend + RSI not overbought + low volatility
        long_signal = kama_up and rsi_ok and vol_ok
        
        # Short: KAMA downtrend + RSI not oversold + low volatility
        short_signal = kama_down and rsi_ok and vol_ok
        
        # Exit when trend changes or volatility spikes
        exit_long = position == 1 and (not kama_up or not rsi_ok or atr_ratio_aligned[i] > 2.0)
        exit_short = position == -1 and (not kama_down or not rsi_ok or atr_ratio_aligned[i] > 2.0)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals