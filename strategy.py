#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a smooth trend line.
In trending markets, price stays above/below KAMA; in ranging markets, price crosses frequently.
Strategy: Go long when price crosses above KAMA with volume confirmation; short when price crosses below.
Use 1-week trend filter (EMA34) to align with higher timeframe direction.
Position size 0.25 to limit trade frequency (~10-20/year) and reduce fee drag.
Works in bull markets via trend continuation and in bear markets via counter-trend reversals at extremes.
"""

name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # Parameters: ER window=10, Fast SC=2/(2+1), Slow SC=2/(30+1)
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [close[0]]  # Initialize with first close
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # Align KAMA to 1d timeframe (already same timeframe, but for consistency)
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}, index=prices.index), kama)
    
    # 1w trend filter: EMA(34) on weekly close
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        if position == 0:
            # LONG: Price crosses above KAMA with volume confirmation and weekly uptrend
            if (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and
                volume_filter[i] and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA with volume confirmation and weekly downtrend
            elif (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and
                  volume_filter[i] and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below KAMA or weekly trend turns down
            if (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]) or \
               (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above KAMA or weekly trend turns up
            if (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]) or \
               (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals