#!/usr/bin/env python3
# 1H_KC_BREAKOUT_4H_TREND_1D_VOLUME_FILTER
# Hypothesis: On 1h timeframe, trade Keltner Channel breakouts only when
# 4h EMA50 trend is aligned and 1d volume is above average.
# Uses volatility-adjusted breakouts to capture trends while avoiding false breakouts in low volatility.
# In bull markets: long when price breaks above KC upper + 4h uptrend + high volume.
# In bear markets: short when price breaks below KC lower + 4h downtrend + high volume.
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years).

name = "1H_KC_BREAKOUT_4H_TREND_1D_VOLUME_FILTER"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # EMA50 for 4h trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Keltner Channel on 1h (20-period EMA, 2*ATR)
    # EMA20 for mid-line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR(20)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # KC Upper and Lower
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need EMA20 and ATR to be ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above KC upper, 4h uptrend, and high volume
            if (close[i] > kc_upper[i] and 
                close[i] > ema50_4h_aligned[i] and 
                volume[i] > vol_avg_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below KC lower, 4h downtrend, and high volume
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume[i] > vol_avg_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA20 (middle of KC) or 4h trend turns down
            if (close[i] < ema20[i] or 
                close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above EMA20 or 4h trend turns up
            if (close[i] > ema20[i] or 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals