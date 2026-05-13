# 4h_WilliamsAlligator_ElderRay_Trend
# Hypothesis: Williams Alligator defines trend direction (jaws/teeth/lips alignment), Elder Ray measures bull/bear power, combined with volume confirmation for high-probability entries on 4h timeframe.
# Long: Alligator bullish (lips > teeth > jaws) + Bull Power > 0 + volume > 1.5x average
# Short: Alligator bearish (lips < teeth < jaws) + Bear Power < 0 + volume > 1.5x average
# Exit when Alligator alignment breaks or Elder Ray power crosses zero.
# Designed for 20-40 trades/year to minimize fee drift. Works in both bull and bear by capturing strong trends with confirmation.

name = "4h_WilliamsAlligator_ElderRay_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Williams Alligator (13,8,5 SMAs shifted)
    # Jaws: 13-period SMA shifted 8 bars
    # Teeth: 8-period SMA shifted 5 bars
    # Lips: 5-period SMA shifted 3 bars
    def sma(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result

    jaws_raw = sma(high, 13)
    teeth_raw = sma(high, 8)
    lips_raw = sma(high, 5)

    # Shift the SMAs
    jaws = np.roll(jaws_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)

    # Invalidate shifted values
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan

    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Alligator bullish + Bull Power positive + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaws[i] and bull_power[i] > 0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish + Bear Power negative + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaws[i] and bear_power[i] < 0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks or Bull Power turns negative
            if not (lips[i] > teeth[i] and teeth[i] > jaws[i]) or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks or Bear Power turns positive
            if not (lips[i] < teeth[i] and teeth[i] < jaws[i]) or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals