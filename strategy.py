#!/usr/bin/env python3
# 12h_KAMA_Trend_With_RSI_and_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing a reliable trend filter in both trending and ranging markets.
# Combined with RSI for momentum and Choppiness Index for regime detection, this strategy avoids false signals.
# Trades only in the direction of the 12h KAMA trend, with RSI avoiding overbought/oversold extremes and chop filter ensuring trades occur in trending regimes.
# Target: 12-37 trades/year to minimize fee drag.

name = "12h_KAMA_Trend_With_RSI_and_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (more stable for 12h strategy)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data for RSI and Chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w KAMA for trend filter
    # KAMA: ER = |Price Change| / Volatility, SC = [ER * (fastest - slowest) + slowest]^2
    close_1w = pd.Series(df_1w['close'])
    change = abs(close_1w.diff(10))  # 10-period change for ER
    volatility = close_1w.diff().abs().rolling(window=10).sum()  # 10-period volatility
    er = change / volatility.replace(0, np.nan)
    fastest = 2 / (2 + 1)  # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    kama = [close_1w.iloc[0]]  # Initialize with first value
    for i in range(1, len(close_1w)):
        if np.isnan(sc.iloc[i]):
            kama.append(kama[-1])
        else:
            kama.append(kama[-1] + sc.iloc[i] * (close_1w.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate daily RSI(14)
    close_1d = pd.Series(df_1d['close'])
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Calculate daily Choppiness Index(14)
    # CHOP = 100 * log10(sum(ATR) / (max_high - min_low)) / log10(period)
    atr_list = []
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).sum()
    max_high = df_1d['high'].rolling(window=14).max()
    min_low = df_1d['low'].rolling(window=14).min()
    chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1w KAMA (30), RSI (14), Chop (14)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1w trend filter
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # RSI filter: avoid extremes
        rsi_ok_long = rsi_aligned[i] < 70  # Not overbought
        rsi_ok_short = rsi_aligned[i] > 30  # Not oversold
        
        # Chop filter: only trade in trending markets (CHOP < 38.2)
        chop_ok = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long entry: uptrend + RSI not overbought + trending regime
            if uptrend and rsi_ok_long and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + RSI not oversold + trending regime
            elif downtrend and rsi_ok_short and chop_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or RSI overbought or choppy market
            if not uptrend or rsi_aligned[i] >= 70 or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or RSI oversold or choppy market
            if not downtrend or rsi_aligned[i] <= 30 or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals