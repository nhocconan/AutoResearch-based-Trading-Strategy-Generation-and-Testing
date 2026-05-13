#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction with RSI(14) mean reversion entries and chop regime filter.
# Enters long when 1d KAMA is rising, RSI(14) < 30 (oversold), and chop regime indicates trending market (CHOP < 38.2).
# Enters short when 1d KAMA is falling, RSI(14) > 70 (overbought), and chop regime indicates trending market (CHOP < 38.2).
# Uses chop regime filter (CHOP(14)) to avoid whipsaws in ranging markets.
# Position size: 0.25 discrete levels to minimize fee churn.
# Designed for low trade frequency (~10-25/year) on 1d timeframe by requiring confluence of trend, momentum extreme, and regime.
# Works in both bull and bear markets: 1d KAMA captures adaptive trend, RSI provides mean-reversion entries within trend,
# chop filter avoids false signals during sideways consolidation.

name = "1d_KAMA_Trend_RSI_MR_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA trend and chop regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d KAMA (adaptive trend) - ER=10, FC=2, SC=30
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series.diff(10))
    volatility = close_1d_series.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/10 - 1/30) + 1/30) ** 2  # smoothing constant
    kama = [np.nan] * len(close_1d_series)
    kama[9] = close_1d_series.iloc[9]  # seed
    for i in range(10, len(close_1d_series)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_1d_series.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama = np.array(kama)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d RSI(14)
    delta = close_1d_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1d Choppy Index (CHOP) - regime filter
    atr_1d = pd.DataFrame({'high': high_1d, 'low': low_1d, 'close': close_1d})
    atr_1d['tr'] = np.maximum(
        atr_1d['high'] - atr_1d['low'],
        np.maximum(
            abs(atr_1d['high'] - atr_1d['close'].shift(1)),
            abs(atr_1d['low'] - atr_1d['close'].shift(1))
        )
    )
    atr_sum = atr_1d['tr'].rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        if chop_aligned[i] >= 38.2:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d KAMA rising, RSI oversold (<30), trending regime
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 30:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d KAMA falling, RSI overbought (>70), trending regime
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1d KAMA (trend change) OR RSI > 70 (overbought)
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1d KAMA (trend change) OR RSI < 30 (oversold)
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals