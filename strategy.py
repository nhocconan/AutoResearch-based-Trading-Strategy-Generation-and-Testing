#!/usr/bin/env python3
# Hypothesis: 1h Bollinger Band mean reversion with 4h EMA50 trend filter and 1d volume spike confirmation.
# Long when price touches lower BB(20,2) AND price < 4h EMA50 (bearish 4h context) AND 1d volume > 1.5x 20-period average.
# Short when price touches upper BB(20,2) AND price > 4h EMA50 (bullish 4h context) AND 1d volume > 1.5x 20-period average.
# Exit when price crosses middle BB(20) line.
# Uses Bollinger Bands for mean reversion in ranging markets, 4h EMA50 to avoid counter-trend trades, and 1d volume to confirm institutional interest.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag on 1h timeframe.

name = "1h_BB_MeanReversion_4hEMA50_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1h Bollinger Bands ---
    bb_period = 20
    bb_std = 2.0
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std * bb_std_dev)
    bb_lower = bb_ma - (bb_std * bb_std_dev)
    bb_middle = bb_ma  # exit signal
    
    # --- 4h EMA50 Trend Filter ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- 1d Volume Spike Confirmation ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike_1d = volume_1d > (1.5 * vol_ma_20_1d_aligned)  # aligned inside align_htf_to_ltf
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):  # start after BB warmup
        # Skip if missing data
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i]) or
            np.isnan(bb_middle[i]) or
            np.isnan(volume_spike_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches lower BB + price < 4h EMA50 (bearish 4h context) + 1d volume spike
            if (low[i] <= bb_lower[i] and 
                close[i] < ema_50_4h_aligned[i] and 
                volume_spike_1d[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price touches upper BB + price > 4h EMA50 (bullish 4h context) + 1d volume spike
            elif (high[i] >= bb_upper[i] and 
                  close[i] > ema_50_4h_aligned[i] and 
                  volume_spike_1d[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above middle BB
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses below middle BB
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals