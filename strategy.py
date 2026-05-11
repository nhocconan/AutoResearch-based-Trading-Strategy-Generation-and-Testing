#!/usr/bin/env python3
name = "4h_ParabolicSAR_VolumeSpike_1dTrend"
timeframe = "4h"
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
    
    # 1d trend: close above/below 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Parabolic SAR calculation
    # Parameters: initial acceleration factor = 0.02, max = 0.2
    psar = np.zeros(n)
    psar[0] = low[0]  # start with first low
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = 0.02
    ep = high[0]  # extreme point
    
    for i in range(1, n):
        if trend == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if psar[i] > low[i]:  # trend reversal
                trend = -1
                psar[i] = ep
                af = 0.02
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, 0.2)
        else:  # downtrend
            psar[i] = psar[i-1] - af * (psar[i-1] - ep)
            if psar[i] < high[i]:  # trend reversal
                trend = 1
                psar[i] = ep
                af = 0.02
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, 0.2)
    
    # Volume filter: volume > 2.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for PSAR and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(psar[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above PSAR + 1d uptrend + volume spike
            if close[i] > psar[i] and close[i-1] <= psar[i-1] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below PSAR + 1d downtrend + volume spike
            elif close[i] < psar[i] and close[i-1] >= psar[i-1] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below PSAR or 1d trend down
            if close[i] < psar[i] and close[i-1] >= psar[i-1] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above PSAR or 1d trend up
            if close[i] > psar[i] and close[i-1] <= psar[i-1] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals