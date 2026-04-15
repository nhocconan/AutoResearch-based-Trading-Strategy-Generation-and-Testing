#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and weekly trend direction
# Long when price breaks above 6h Donchian upper + 1d ATR ratio < 0.8 (low vol) + 1w close > 1w EMA34
# Short when price breaks below 6h Donchian lower + 1d ATR ratio < 0.8 + 1w close < 1w EMA34
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# ATR filter targets expansion after compression, effective in both bull and bear markets.
# Weekly EMA34 provides major trend filter reducing whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: ATR(14) and ATR(50) ratio ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # Current volatility vs longer-term volatility
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === 1w Indicator: EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 6h Donchian Channel (20-period) ===
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 34) + 5  # ATR(50) + Donchian(20) + EMA34(1w) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: looking for low volatility contraction (ratio < 0.8)
        vol_filter = atr_ratio_aligned[i] < 0.8
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 6h Donchian upper (close > upper)
        # 2. Low volatility environment (ATR ratio < 0.8)
        # 3. Weekly uptrend (close > weekly EMA34)
        if (close[i] > donchian_upper[i]) and \
           vol_filter and \
           (close[i] > ema_34_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 6h Donchian lower (close < lower)
        # 2. Low volatility environment (ATR ratio < 0.8)
        # 3. Weekly downtrend (close < weekly EMA34)
        elif (close[i] < donchian_lower[i]) and \
             vol_filter and \
             (close[i] < ema_34_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_1dATRratio_1wEMA34_v1"
timeframe = "6h"
leverage = 1.0