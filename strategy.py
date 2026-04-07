#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover (12/26) with 4h trend filter and daily volatility filter
# Uses EMA(12)/EMA(26) crossover for entry timing on 1h, 4h EMA(50) for trend direction,
# and daily ATR(14) for volatility filtering to avoid choppy markets.
# Designed for low trade frequency (target: 15-37 trades/year) by requiring
# alignment of 1h momentum with 4h trend and low volatility regime.
# Works in bull markets via trend-following EMA crossovers and in bear markets
# by filtering trades to only those aligned with higher timeframe trend.

name = "ema_crossover_4h_trend_vol_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Daily data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # EMA(12) and EMA(26) for 1h momentum
    ema12 = pd.Series(close).ewm(span=12, adjust=False).values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).values
    
    # 4h EMA(50) for trend direction
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False).values
    atr_ma_50 = pd.Series(atr_14).ewm(span=50, adjust=False).values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema12[i]) or np.isnan(ema26[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_ma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # EMA crossover signal
        bullish_cross = ema12[i] > ema26[i]
        bearish_cross = ema12[i] < ema26[i]
        
        # 4h trend filter
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        
        # Daily volatility filter: only trade when volatility is below average
        low_volatility = atr_14[i] < atr_ma_50_aligned[i]
        
        # Entry conditions: EMA crossover aligned with 4h trend and low volatility
        if bullish_cross and uptrend_4h and low_volatility:
            signals[i] = 0.20
        elif bearish_cross and downtrend_4h and low_volatility:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals