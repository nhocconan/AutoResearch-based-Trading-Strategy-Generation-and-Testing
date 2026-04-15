#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction with RSI and weekly volatility regime filter
# KAMA adapts to market efficiency, providing smooth trend direction.
# RSI(2) identifies short-term extremes within the trend for mean-reversion entries.
# Weekly ATR-based regime filter avoids trading in extreme volatility regimes.
# Designed to work in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA(10,2,30) - adaptive trend
    change = np.abs(np.diff(close, k=10))
    abs_change = np.abs(np.diff(close, k=1))
    er = np.zeros(n)
    er[10:] = change[10:] / (np.abs(np.diff(close, k=1)).rolling(10).sum()[10:] + 1e-10)
    sc = (er * 0.6 + 0.06) ** 2
    kama = np.full(n, np.nan)
    kama[9] = close[9]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(2) for short-term extremes
    delta = np.diff(close)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(2, min_periods=2).mean()
    roll_down = pd.Series(down).rolling(2, min_periods=2).mean()
    rs = roll_up / (roll_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1w, atr_20)
    atr_median = pd.Series(atr_20_aligned).rolling(window=50, min_periods=50).median()
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr_20_aligned[i]) or np.isnan(atr_median[i])):
            continue
        
        # Regime filter: avoid extreme volatility
        vol_regime = (atr_20_aligned[i] > 0.5 * atr_median[i]) and (atr_20_aligned[i] < 2.0 * atr_median[i])
        
        # Long: price > KAMA (uptrend) + RSI < 10 (oversold) + vol regime
        if (close[i] > kama[i] and 
            rsi[i] < 10 and 
            vol_regime):
            signals[i] = 0.25
        
        # Short: price < KAMA (downtrend) + RSI > 90 (overbought) + vol regime
        elif (close[i] < kama[i] and 
              rsi[i] > 90 and 
              vol_regime):
            signals[i] = -0.25
        
        # Exit: RSI returns to neutral zone (40-60)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and rsi[i] >= 40) or
               (signals[i-1] == -0.25 and rsi[i] <= 60))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_KAMA_RSI_VolatilityRegime"
timeframe = "1d"
leverage = 1.0