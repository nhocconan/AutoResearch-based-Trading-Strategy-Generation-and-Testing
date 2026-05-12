# Solution: 4h_TradingView_AVWAP_Crossover_Trend_Filter
# Hypothesis: Anchored Volume Weighted Average Price (VWAP) crossovers with 1d EMA50 trend filter
# and volume confirmation capture institutional momentum. Anchors reset at daily/weekly starts.
# AVWAP acts as dynamic support/resistance, reducing whipsaws in choppy markets.
# Target: 25-40 trades/year per symbol with disciplined risk management.

name = "4h_TradingView_AVWAP_Crossover_Trend_Filter"
timeframe = "4h"
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
    open_time = pd.to_datetime(prices['open_time'])

    # Get 1d data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate typical price and VWAP components
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume

    # Initialize AVWAP arrays with daily anchors
    avwap = np.full(n, np.nan)
    cum_tpv = 0.0
    cum_vol = 0.0

    # Calculate AVWAP with daily resets
    for i in range(n):
        # Reset at start of each day (00:00 UTC)
        if i == 0 or open_time[i].date() != open_time[i-1].date():
            cum_tpv = 0.0
            cum_vol = 0.0
        cum_tpv += tpv[i]
        cum_vol += volume[i]
        if cum_vol > 0:
            avwap[i] = cum_tpv / cum_vol

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if np.isnan(avwap[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close crosses above AVWAP + 1d uptrend + volume confirmation
            if close[i] > avwap[i] and close[i-1] <= avwap[i-1] and close[i] > ema50_1d_aligned[i] and volume[i] > np.median(volume[max(0, i-20):i+1]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close crosses below AVWAP + 1d downtrend + volume confirmation
            elif close[i] < avwap[i] and close[i-1] >= avwap[i-1] and close[i] < ema50_1d_aligned[i] and volume[i] > np.median(volume[max(0, i-20):i+1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below AVWAP or 1d trend turns down
            if close[i] < avwap[i] and close[i-1] >= avwap[i-1] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above AVWAP or 1d trend turns up
            if close[i] > avwap[i] and close[i-1] <= avwap[i-1] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals