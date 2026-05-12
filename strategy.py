#/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeFilter
Hypothesis: Combines intraday mean reversion with multi-timeframe trend and volume confirmation.
Uses 1h price action for entry timing (break of Camarilla R1/S1 levels), 4h for trend direction (price above/below EMA50),
and 1d volume spike to confirm institutional participation. Designed for 15-35 trades/year to minimize fee drag
while capturing breakouts in both bull and bear markets. The 1d volume filter reduces false breakouts during low
liquidity periods, and the session filter (08-20 UTC) avoids Asian session noise.
"""

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Get 1d data for Camarilla levels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Use previous day's OHLC for Camarilla calculation (avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12

    # Align Camarilla levels to 1h timeframe with 1-day delay
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)

    # Calculate 1d volume average (20-period) for spike detection
    vol_avg_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d, additional_delay_bars=1)

    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema50_val = ema50_4h_aligned[i]
        vol_avg_val = vol_avg_20_1d_aligned[i]
        vol_1d_val = df_1d['volume'].values[i // 24]  # 24x 1h bars in 1d

        if np.isnan(camarilla_r1_val) or np.isnan(camarilla_s1_val) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 + 4h uptrend + volume spike
            if close[i] > camarilla_r1_val and close[i] > ema50_val and vol_1d_val > vol_avg_val * 1.8:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 + 4h downtrend + volume spike
            elif close[i] < camarilla_s1_val and close[i] < ema50_val and vol_1d_val > vol_avg_val * 1.8:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below S1 or trend reversal
            if close[i] < camarilla_s1_val or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price above R1 or trend reversal
            if close[i] > camarilla_r1_val or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals