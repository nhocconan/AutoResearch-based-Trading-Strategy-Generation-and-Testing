# [12h_Williams_Alligator_Elder_Ray] Williams Alligator for trend direction + Elder Ray for momentum + volume confirmation on 12h
# Works in bull/bear: Alligator identifies trend, Elder Ray confirms momentum, volume filters false signals
# Target: 25-35 trades/year per symbol to minimize fee drag

name = "12h_Williams_Alligator_Elder_Ray"
timeframe = "12h"
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

    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator on 12h: SMAs with specific periods
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period SMA shifted 8
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period SMA shifted 5
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period SMA shifted 3
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + Weekly trend up + Volume confirm
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                close[i] > ema50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Jaw > Teeth > Lips (bearish alignment) + Bear Power < 0 + Weekly trend down + Volume confirm
            elif (jaw[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish OR Bear Power negative OR Weekly trend breaks
            if (jaw[i] > teeth[i] or bear_power[i] < 0 or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish OR Bull Power positive OR Weekly trend breaks
            if (lips[i] > teeth[i] or bull_power[i] > 0 or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals