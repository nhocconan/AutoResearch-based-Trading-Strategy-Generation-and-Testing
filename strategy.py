#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Chop
# Hypothesis: KAMA identifies the primary trend direction on 12h chart. 
# RSI(14) provides entry timing with oversold/overbought conditions. 
# Choppiness index (CHOP) filters trades: only trade in trending markets (CHOP < 38.2) 
# to avoid whipsaws in sideways markets. Designed for low-frequency, high-conviction trades
# that work in both bull and bear markets by following the trend.

name = "12h_KAMA_Trend_RSI_Chop"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for higher timeframe trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA trend: faster adaptation to trend changes
    close_1d = df_1d['close']
    delta = close_1d.diff().abs()
    vol = delta.rolling(window=10, min_periods=10).sum()
    er = delta / vol.replace(0, 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [close_1d.iloc[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d Choppiness Index: identifies trending vs ranging markets
    atr_1d = []
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).sum()
    sum_tr14 = atr
    hh = df_1d['high'].rolling(window=14, min_periods=14).max()
    ll = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_tr14 / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend condition: price vs KAMA on 1d
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        
        # Chop filter: only trade in trending markets (CHOP < 38.2)
        trending_market = chop_1d_aligned[i] < 38.2
        
        # RSI conditions for entry
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70

        if position == 0:
            # LONG: Price above KAMA (uptrend) + trending market + RSI oversold
            if price_above_kama and trending_market and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + trending market + RSI overbought
            elif price_below_kama and trending_market and rsi_overbought:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA OR choppy market (CHOP > 61.8)
            if price_below_kama or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR choppy market (CHOP > 61.8)
            if price_above_kama or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals