#!/usr/bin/env python3
# Hypothesis: 1h timeframe with 4h RSI(14) trend filter and 1d volume spike confirmation.
# In strong trends (4h RSI > 60 or < 40), price tends to continue in the direction of the trend.
# Enters long when 4h RSI > 60 + 1d volume > 1.5x 20-period average + price pulls back to EMA(20) on 1h.
# Enters short when 4h RSI < 40 + 1d volume > 1.5x 20-period average + price rallies to EMA(20) on 1h.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods.
# Target: 60-150 total trades over 4 years (15-37/year) with size 0.20.

name = "1h_RSITrend_VolumeSpike_EMA_Pullback"
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
    
    # 4h RSI(14) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close']
    delta = close_4h.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_values = rsi_4h.fillna(50).values  # fill neutral for warmup
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)
    
    # 1d volume spike: current volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume']
    vol_ma_20 = volume_1d.rolling(window=20, min_periods=20).mean()
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_values = vol_spike.fillna(False).values
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_values)
    
    # 1h EMA(20) for pullback entry
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 4h RSI > 60 (uptrend) + volume spike + price pulls back to EMA(20) from below
            if (rsi_4h_aligned[i] > 60 and
                vol_spike_aligned[i] and
                close[i] >= ema_20[i] and
                close[i-1] < ema_20[i-1]):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h RSI < 40 (downtrend) + volume spike + price rallies to EMA(20) from above
            elif (rsi_4h_aligned[i] < 40 and
                  vol_spike_aligned[i] and
                  close[i] <= ema_20[i] and
                  close[i-1] > ema_20[i-1]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI turns neutral (40-60) or price breaks above EMA(20) with momentum
            if (rsi_4h_aligned[i] < 40 or  # trend reversal
                close[i] > ema_20[i] * 1.01):  # extended move
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI turns neutral (40-60) or price breaks below EMA(20) with momentum
            if (rsi_4h_aligned[i] > 60 or  # trend reversal
                close[i] < ema_20[i] * 0.99):  # extended move
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals