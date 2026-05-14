#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-day high AND 1w EMA200 is bullish AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below 20-day low AND 1w EMA200 is bearish AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the 10-day EMA.
# Uses discrete position sizing (0.30) to limit fee churn. Designed for BTC/ETH robustness by capturing strong breakouts with volume confirmation in trending markets.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Donchian20_Breakout_1wEMA200_1dVolumeSpike_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA200 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian high AND 1w EMA200 is bullish (price > EMA200) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below Donchian low AND 1w EMA200 is bearish (price < EMA200) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to 10-day EMA
            if close[i] <= ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price retraces to 10-day EMA
            if close[i] >= ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals