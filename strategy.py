#!/usr/bin/env python3

# 1d_1W_Camarilla_Pivot_Breakout_Trend_Volume
# Hypothesis: Breakout above/below Camarilla R4/S4 levels on 1d with 1w trend filter and volume confirmation.
# Uses weekly timeframe for trend direction and daily for entry/exit to reduce trade frequency.
# Combines price channel breakout with trend alignment and volume confirmation to avoid false signals.
# Targets 15-25 trades per year to minimize fee drag while capturing significant moves.

name = "1d_1W_Camarilla_Pivot_Breakout_Trend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Calculate daily Camarilla levels
    # Using previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first values to current values to avoid NaN
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla calculation
    range_val = prev_high - prev_low
    camarilla_r4 = prev_close + (1.1 * range_val * 1.1 / 2)
    camarilla_s4 = prev_close - (1.1 * range_val * 1.1 / 2)

    # Volume confirmation: current volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_r4[i]) or
            np.isnan(camarilla_s4[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1w
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R4 with bullish trend and volume confirmation
            if close[i] > camarilla_r4[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S4 with bearish trend and volume confirmation
            elif close[i] < camarilla_s4[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla C (midpoint) or trend turns bearish
            camarilla_c = (prev_high[i] + prev_low[i]) / 2  # Camarilla pivot point
            if close[i] < camarilla_c or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla C or trend turns bullish
            camarilla_c = (prev_high[i] + prev_low[i]) / 2
            if close[i] > camarilla_c or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals