# US Dollar Index (DXY) as a Safe Haven Proxy for Crypto Risk-Off Moves
# Hypothesis: DXY strength (risk-off) precedes crypto selloffs; DXY weakness (risk-on) precedes crypto rallies.
# Use 1d DXY as a leading indicator for 1h BTC/ETH/SOL trends. Enter long when DXY is falling and BTC is above its 1h EMA20 with volume confirmation.
# Enter short when DXY is rising and BTC is below its 1h EMA20 with volume confirmation.
# Exit on mean reversion to EMA20 or DXY reversal.
# Timeframe: 1h (entry timing), DXY as 1d trend filter.
# Target: 15-30 trades/year to avoid fee drain.

name = "DXY_Risk_On_Off_1H_Trend"
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

    # Load 1h EMA20 for trend and mean reversion exit
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Volume filter: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Load daily DXY data as risk-on/risk-off proxy
    # Note: We assume the 'prices' DataFrame includes a 'dxy_close' column from external data merge
    # In practice, this would be merged before calling generate_signals; if not present, skip DXY filter
    if 'dxy_close' in prices.columns:
        dxy_close = prices['dxy_close'].values
        # Daily DXY trend: 20-period EMA on daily data, but we only have hourly prices
        # Instead, use 24-period EMA on hourly DXY as proxy for daily trend (24h = 1 day)
        dxy_ema24 = pd.Series(dxy_close).ewm(span=24, adjust=False, min_periods=24).mean().values
        # DXY rising = risk-off (bad for crypto), DXY falling = risk-on (good for crypto)
        dxy_rising = dxy_close > dxy_ema24   # DXY above its trend = strengthening
        dxy_falling = dxy_close < dxy_ema24  # DXY below its trend = weakening
    else:
        # If DXY data not available, neutral stance (do not filter)
        dxy_rising = np.zeros(n, dtype=bool)
        dxy_falling = np.zeros(n, dtype=bool)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):  # Start after DXY EMA warmup
        # Skip if any required value is NaN
        if (np.isnan(ema20[i]) or np.isnan(vol_avg_20[i]) or
            (np.isnan(dxy_close[i]) if 'dxy_close' in prices.columns else False)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        vol_ok = volume[i] > vol_avg_20[i] * 1.5

        if position == 0:
            # LONG: DXY falling (risk-on) + price above EMA20 + volume
            if dxy_falling[i] and close[i] > ema20[i] and vol_ok:
                signals[i] = 0.20
                position = 1
            # SHORT: DXY rising (risk-off) + price below EMA20 + volume
            elif dxy_rising[i] and close[i] < ema20[i] and vol_ok:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA20 (mean reversion) OR DXY turns rising (risk-off)
            if close[i] < ema20[i] or dxy_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA20 (mean reversion) OR DXY turns falling (risk-on)
            if close[i] > ema20[i] or dxy_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals