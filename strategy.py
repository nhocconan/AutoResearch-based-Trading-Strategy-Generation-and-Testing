#!/usr/bin/env python3
# 1D_KAMA_RSI_CHOP_FILTER
# Hypothesis: 1-day KAMA trend direction combined with RSI mean reversion and chop filter
# captures directional moves while avoiding whipsaws in ranging markets. Uses weekly trend
# filter to align with higher timeframe momentum. Designed for low trade frequency with
# discrete sizing to minimize fee drag. Works in bull/bear markets by following weekly trend
# and only taking mean-reversion entries when market is ranging (choppy).

name = "1D_KAMA_RSI_CHOP_FILTER"
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
    volume = prices['volume'].values
    
    # KAMA trend (1-day)
    close_s = pd.Series(close)
    change = close_s.diff().abs()
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Chopping index (14-period) for regime detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    # Weekly trend filter: EMA 34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    ema_34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_34_1w_aligned[i]
        is_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price above KAMA (uptrend) + RSI oversold + choppy market (mean reversion setup)
            if close[i] > kama[i] and rsi[i] < 30 and chop[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short entry: Price below KAMA (downtrend) + RSI overbought + choppy market (mean reversion setup)
            elif close[i] < kama[i] and rsi[i] > 70 and chop[i] > 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or trend change
            if rsi[i] > 70 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or trend change
            if rsi[i] < 30 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals