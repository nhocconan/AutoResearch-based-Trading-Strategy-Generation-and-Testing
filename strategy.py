# 1h_VolumeBreakout_4hTrend
# Hypothesis: On 1h, volume surges (>1.5x 20-period average) combined with price breaking above/below 4h VWAP bands capture momentum. 4h trend (EMA50) filters direction to avoid counter-trend trades. Session filter (08-20 UTC) reduces noise. Targets 15-30 trades/year with low turnover.

name = "1h_VolumeBreakout_4hTrend"
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

    # Get 4h data (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values

    # Calculate 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Calculate 4h VWAP and bands (typical price * volume)
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    vwap_num = (typical_price_4h * volume_4h).cumsum()
    vwap_den = volume_4h.cumsum()
    vwap_4h = vwap_num / vwap_den
    # Avoid division by zero
    vwap_4h = np.where(vwap_den == 0, np.nan, vwap_4h)
    # Standard deviation of typical price from VWAP
    dev = typical_price_4h - vwap_4h
    var = (dev * dev * volume_4h).cumsum() / vwap_den
    vwap_std_4h = np.sqrt(var)
    vwap_std_4h = np.where(vwap_den == 0, np.nan, vwap_std_4h)
    # Upper and lower bands (2 std dev)
    upper_band_4h = vwap_4h + 2 * vwap_std_4h
    lower_band_4h = vwap_4h - 2 * vwap_std_4h
    upper_band_aligned = align_htf_to_ltf(prices, df_4h, upper_band_4h)
    lower_band_aligned = align_htf_to_ltf(prices, df_4h, lower_band_4h)

    # Volume confirmation: 1.5x 20-period average on 1h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Session filter: 08-20 UTC (precompute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Get aligned values for current 1h bar
        ema50 = ema50_4h_aligned[i]
        upper_band = upper_band_aligned[i]
        lower_band = lower_band_aligned[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema50) or np.isnan(upper_band) or 
            np.isnan(lower_band) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper band + volume surge + price above 4h EMA50
            if (close[i] > upper_band and 
                volume[i] > vol_avg_val * 1.5 and 
                close[i] > ema50):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below lower band + volume surge + price below 4h EMA50
            elif (close[i] < lower_band and 
                  volume[i] > vol_avg_val * 1.5 and 
                  close[i] < ema50):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower band or price below 4h EMA50
            if (close[i] < lower_band or close[i] < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above upper band or price above 4h EMA50
            if (close[i] > upper_band or close[i] > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals