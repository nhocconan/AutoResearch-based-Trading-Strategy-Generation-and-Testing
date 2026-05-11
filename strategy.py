#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_1dTrend"
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
    volume = prices['volume'].values
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # TRIX(9) on 4h: triple EMA of log returns
    log_returns = np.log(close / np.roll(close, 1))
    log_returns[0] = 0  # first value has no previous
    ema1 = pd.Series(log_returns).ewm(span=9, adjust=False, min_periods=9).mean()
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean()
    trix = ema3.values * 100  # scale for readability
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for TRIX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix[i]) or np.isnan(ema34_1d[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(trend_up_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX > 0 (bullish momentum) + daily uptrend + volume spike
            if trix[i] > 0 and trend_up_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX < 0 (bearish momentum) + daily downtrend + volume spike
            elif trix[i] < 0 and not trend_up_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative OR daily trend turns down
            if trix[i] < 0 or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive OR daily trend turns up
            if trix[i] > 0 or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals