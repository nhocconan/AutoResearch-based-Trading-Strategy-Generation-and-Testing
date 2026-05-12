# 6h_ADX_Trend_Strength_With_Volume_Spike
# Hypothesis: Combine ADX trend strength with volume spike on 6h timeframe. ADX > 25 indicates strong trend, while volume > 2x average confirms momentum.
# Enter long when price > EMA20, ADX > 25, and volume spike; short when price < EMA20, ADX > 25, and volume spike.
# Exit when trend weakens (ADX < 20) or price crosses EMA20 in opposite direction.
# Designed for 12-30 trades/year per symbol, works in both bull and bear via ADX trend filter.

name = "6h_ADX_Trend_Strength_With_Volume_Spike"
timeframe = "6h"
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

    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth TR, +DM, -DM
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx[np.isnan(dx) | np.isinf(dx)] = 0
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx

    # Calculate ADX
    adx = calculate_adx(high, low, close, 14)

    # EMA20 for trend direction
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume spike: current volume > 2x average of last 6 periods (6 hours)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend strength filter
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20

        if position == 0:
            # LONG: Price above EMA20, strong trend, volume spike
            if close[i] > ema_20[i] and strong_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below EMA20, strong trend, volume spike
            elif close[i] < ema_20[i] and strong_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakens or price crosses below EMA20
            if weak_trend or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens or price crosses above EMA20
            if weak_trend or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals