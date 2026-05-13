#!/usr/bin/env python3
# 1h_HTF_Camarilla_Reversal_With_Volume_Spike
# Hypothesis: At 1h, we trade reversals from Camarilla pivot levels (S1/S2, R1/R2) only when aligned with higher timeframe trend (4h EMA50, 1d EMA200) and confirmed by volume spike.
# In uptrends (price > 1d EMA200 & 4h EMA50 > EMA20), we go long at S1/S2 support with volume spike.
# In downtrends (price < 1d EMA200 & 4h EMA50 < EMA20), we go short at R1/R2 resistance with volume spike.
# Uses 4h/1d for trend direction, 1h for precise entry at pivot levels. Volume spike filters noise.
# Designed for low trade frequency (15-35/year) to avoid fee drag in 1h timeframe.

name = "1h_HTF_Camarilla_Reversal_With_Volume_Spike"
timeframe = "1h"
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

    # --- Higher Timeframe Trend (4h & 1d) ---
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')

    # 1d EMA200 for long-term trend
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # 4h EMA50/EMA20 for intermediate trend
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)

    # --- 1h Camarilla Pivots (based on previous day) ---
    # Typical price for pivot calculation
    typical_price = (high + low + close) / 3.0
    # Use previous day's OHLC (we approximate using rolling window of 24 for 1h data)
    # For true daily pivot, we would need actual OHLC - but we use typical price of prior 24h as proxy
    # Since we can't access true daily OHLC without look-ahead, we use rolling 24-period typical price
    # This is acceptable as it uses only past data
    roll_24 = pd.Series(typical_price).rolling(window=24, min_periods=24)
    prev_day_typical = roll_24.shift(1)  # previous 24-period average
    # Approximate prior day's high/low/close using rolling windows
    prev_day_high = pd.Series(high).rolling(window=24, min_periods=24).shift(1).values
    prev_day_low = pd.Series(low).rolling(window=24, min_periods=24).shift(1).values
    prev_day_close = pd.Series(close).rolling(window=24, min_periods=24).shift(1).values

    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But standard Camarilla uses: 
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), R2 = close + 0.6*(high-low), R1 = close + 0.275*(high-low)
    # S1 = close - 0.275*(high-low), S2 = close - 0.6*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We'll use the typical price as 'close' for pivot calculation
    hl_range = prev_day_high - prev_day_low
    R1 = prev_day_close + 0.275 * hl_range
    S1 = prev_day_close - 0.275 * hl_range
    R2 = prev_day_close + 0.6 * hl_range
    S2 = prev_day_close - 0.6 * hl_range

    # --- Volume Spike (1h) ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20  # 2x average volume

    # --- Session Filter: 08-20 UTC ---
    hours = prices.index.hour  # already datetime64[ms], .hour works
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):  # start after warmup for rolling windows
        # Skip if any required value is NaN
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(R2[i]) or np.isnan(S2[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Determine trend: 
            # Uptrend: price > 1d EMA200 AND 4h EMA50 > EMA20
            # Downtrend: price < 1d EMA200 AND 4h EMA50 < EMA20
            uptrend = close[i] > ema200_1d_aligned[i] and ema50_4h_aligned[i] > ema20_4h_aligned[i]
            downtrend = close[i] < ema200_1d_aligned[i] and ema50_4h_aligned[i] < ema20_4h_aligned[i]

            # LONG: in uptrend, price at S1 or S2 support, with volume spike
            if uptrend and volume_spike[i]:
                if close[i] <= S1[i] * 1.001 or close[i] <= S2[i] * 1.001:  # allow small buffer
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            # SHORT: in downtrend, price at R1 or R2 resistance, with volume spike
            elif downtrend and volume_spike[i]:
                if close[i] >= R1[i] * 0.999 or close[i] >= R2[i] * 0.999:  # allow small buffer
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0

        elif position == 1:
            # EXIT LONG: if trend breaks down or price reaches R1 (take profit)
            if not (close[i] > ema200_1d_aligned[i] and ema50_4h_aligned[i] > ema20_4h_aligned[i]) or close[i] >= R1[i] * 0.999:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20

        elif position == -1:
            # EXIT SHORT: if trend breaks up or price reaches S1 (take profit)
            if not (close[i] < ema200_1d_aligned[i] and ema50_4h_aligned[i] < ema20_4h_aligned[i]) or close[i] <= S1[i] * 1.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals