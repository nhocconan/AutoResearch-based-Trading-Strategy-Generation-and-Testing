#164659 - 6h_Camarilla_R3_S3_Fade_1dTrend_Volume
# Hypothesis: Fade at Camarilla R3/S3 levels with 1d trend filter and volume confirmation.
# In ranging markets, price reverses at R3/S3; in trending markets, 1d trend filter avoids counter-trend fades.
# Target: 15-30 trades/year per symbol. Works in bull/bear via trend filter.

name = "6h_Camarilla_R3_S3_Fade_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Use previous day's typical price for calculation
    prev_typical = np.concatenate([[typical_price[0]], typical_price[:-1]])
    # Daily range from previous day
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    range_prev = prev_high - prev_low
    
    # Camarilla levels
    R3 = prev_typical + range_prev * 1.1 / 4
    S3 = prev_typical - range_prev * 1.1 / 4
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Fade at R3/S3 with 1d trend filter
        if position == 0:
            # LONG: price at S3, 1d uptrend, volume confirmation
            if close[i] <= S3[i] and uptrend_1d_aligned[i] and volume_conf[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price at R3, 1d downtrend, volume confirmation
            elif close[i] >= R3[i] and downtrend_1d_aligned[i] and volume_conf[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price at typical price or 1d trend turns down
            if close[i] >= prev_typical[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price at typical price or 1d trend turns up
            if close[i] <= prev_typical[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals