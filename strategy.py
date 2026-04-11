#!/usr/bin/env python3
# 12h_1d_cci_volatility_breakout_v1
# Strategy: 12h CCI breakout with volatility filter and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions. Breakouts above +100 or below -100
# signal strong momentum. Combined with volatility filter (low ATR ratio) to avoid false breakouts
# and 1d EMA50 trend filter to align with higher timeframe trend. Designed for low frequency
# (15-25 trades/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_volatility_breakout_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (low volatility filter)
    atr_series = pd.Series(atr)
    atr_avg_50 = atr_series.rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_avg_50  # < 1 = low volatility, > 1 = high volatility
    
    # CCI calculation: (Typical Price - SMA) / (0.015 * Mean Deviation)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    sma_20 = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp_series - sma_20) / (0.015 * mad)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cci.iloc[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(atr_avg_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volatility filter: low volatility environment (ATR ratio < 0.8)
        vol_filter = atr_ratio[i] < 0.8
        
        # Entry logic: CCI breakout + volatility filter + trend alignment
        if (cci.iloc[i] > 100 and  # Strong uptrend breakout
            vol_filter and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (cci.iloc[i] < -100 and  # Strong downtrend breakout
              vol_filter and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral zone or trend change or high volatility
        elif position == 1 and (cci.iloc[i] <= 0 or not uptrend or not vol_filter):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci.iloc[i] >= 0 or not downtrend or not vol_filter):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals