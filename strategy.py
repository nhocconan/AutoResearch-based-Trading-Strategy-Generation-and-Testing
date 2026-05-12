#!/usr/bin/env python3
"""
4h_RSI200_Trend_Breakout_With_Volume
Hypothesis: On 4h timeframe, buy when RSI(200) > 50 and price breaks above Donchian(20) high with volume > 1.5x 20-period average; sell when RSI(200) < 50 and price breaks below Donchian(20) low with volume > 1.5x average. 
Use 1d EMA50 trend filter: only long when price > EMA50, short when price < EMA50. 
Add 1d Bollinger Band width < 50th percentile to avoid choppy regimes. 
Exit when price crosses Donchian midpoint or RSI(200) crosses 50 in opposite direction. 
Target 20-50 trades per year with strong trend filtration to reduce false signals.
"""

name = "4h_RSI200_Trend_Breakout_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 1d Bollinger Band width (20, 2) for regime filter
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma20_1d
    bb_width_rank = pd.Series(bb_width_1d).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_rank_aligned = align_htf_to_ltf(prices, df_1d, bb_width_rank)

    # Calculate RSI(200) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/200, adjust=False, min_periods=200).mean()
    avg_loss = loss.ewm(alpha=1/200, adjust=False, min_periods=200).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when insufficient data

    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Get aligned values for current 4h bar
        ema50 = ema50_1d_aligned[i]
        bb_rank = bb_width_rank_aligned[i]
        rsi_val = rsi[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        donch_mid = donchian_mid[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(bb_rank) or 
            np.isnan(rsi_val) or np.isnan(donch_high) or 
            np.isnan(donch_low) or np.isnan(donch_mid)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: only trade when BB width is in lower 50% (contraction/low volatility)
        if bb_rank > 0.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 50, price breaks above Donchian high, price > EMA50, volume surge
            if (rsi_val > 50 and 
                close[i] > donch_high and 
                close[i] > ema50 and 
                volume[i] > np.nanmean(volume[max(0, i-20):i+1]) * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 50, price breaks below Donchian low, price < EMA50, volume surge
            elif (rsi_val < 50 and 
                  close[i] < donch_low and 
                  close[i] < ema50 and 
                  volume[i] > np.nanmean(volume[max(0, i-20):i+1]) * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below Donchian mid OR RSI crosses below 50
            if (close[i] < donch_mid or rsi_val < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Donchian mid OR RSI crosses above 50
            if (close[i] > donch_mid or rsi_val > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals