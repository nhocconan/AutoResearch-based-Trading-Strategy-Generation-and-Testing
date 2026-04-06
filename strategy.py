#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
In trending markets (Chop < 38.2), follow KAMA direction with RSI filter for momentum.
In ranging markets (Chop > 61.8), mean-revert at Bollinger Bands.
Uses 1w trend filter to avoid counter-trend trades in strong trends.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for indicators (already 1d, but using for consistency)
    df_1d = prices.copy()
    
    # Load 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Efficiency Ratio
    change = abs(df_1d['close'].diff(er_length))
    volatility = abs(df_1d['close'].diff()).rolling(window=er_length, min_periods=er_length).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.full(n, np.nan)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # RSI
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # Chop Index
    atr_period = 14
    chop_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    max_atr = atr.rolling(window=chop_period, min_periods=chop_period).max()
    min_atr = atr.rolling(window=chop_period, min_periods=chop_period).min()
    chop = 100 * np.log10((max_atr - min_atr) / atr) / np.log10(chop_period)
    chop = chop.fillna(50)
    
    # Bollinger Bands for mean reversion
    bb_length = 20
    bb_mult = 2.0
    bb_basis = df_1d['close'].rolling(window=bb_length, min_periods=bb_length).mean()
    bb_dev = bb_mult * df_1d['close'].rolling(window=bb_length, min_periods=bb_length).std()
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
    # 1w EMA for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter
    vol_ma = df_1d['volume'].rolling(window=20, min_periods=20).mean()
    vol_filter = df_1d['volume'] > (0.7 * vol_ma)
    
    # ATR for stoploss
    tr = pd.concat([
        df_1d['high'] - df_1d['low'],
        abs(df_1d['high'] - df_1d['close'].shift()),
        abs(df_1d['low'] - df_1d['close'].shift())
    ], axis=1).max(axis=1)
    atr_sl = tr.rolling(window=15, min_periods=15).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(er_length, 14, bb_length, chop_period) + 5
    
    for i in range(start, n):
        if (np.isnan(kama.iloc[i]) or np.isnan(rsi.iloc[i]) or np.isnan(chop.iloc[i]) or
            np.isnan(bb_upper.iloc[i]) or np.isnan(bb_lower.iloc[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma.iloc[i]) or np.isnan(atr_sl.iloc[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA
        uptrend_1w = df_1d['close'].iloc[i] > ema_1w_aligned[i]
        downtrend_1w = df_1d['close'].iloc[i] < ema_1w_aligned[i]
        
        if position == 1:  # long position
            # Exit: KAMA reversal OR RSI overbought OR stoploss
            if (df_1d['close'].iloc[i] < kama.iloc[i] or
                rsi.iloc[i] > 70 or
                df_1d['close'].iloc[i] <= entry_price - 2.5 * atr_sl.iloc[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: KAMA reversal OR RSI oversold OR stoploss
            if (df_1d['close'].iloc[i] > kama.iloc[i] or
                rsi.iloc[i] < 30 or
                df_1d['close'].iloc[i] >= entry_price + 2.5 * atr_sl.iloc[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if chop.iloc[i] < 38.2:  # Trending market
                # Follow KAMA direction with RSI filter
                long_setup = (df_1d['close'].iloc[i] > kama.iloc[i] and
                              rsi.iloc[i] > 50 and rsi.iloc[i] < 70 and
                              uptrend_1w and vol_filter.iloc[i])
                short_setup = (df_1d['close'].iloc[i] < kama.iloc[i] and
                               rsi.iloc[i] < 50 and rsi.iloc[i] > 30 and
                               downtrend_1w and vol_filter.iloc[i])
            elif chop.iloc[i] > 61.8:  # Ranging market
                # Mean revert at Bollinger Bands
                long_setup = (df_1d['close'].iloc[i] <= bb_lower.iloc[i] and
                              rsi.iloc[i] < 35 and vol_filter.iloc[i])
                short_setup = (df_1d['close'].iloc[i] >= bb_upper.iloc[i] and
                               rsi.iloc[i] > 65 and vol_filter.iloc[i])
            else:  # Transition zone - no trade
                long_setup = False
                short_setup = False
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = df_1d['close'].iloc[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = df_1d['close'].iloc[i]
            else:
                signals[i] = 0.0
    
    return signals