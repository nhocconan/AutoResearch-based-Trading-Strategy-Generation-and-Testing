#!/usr/bin/env python3
# Hypothesis: 1h mean reversion with 4h Donchian breakout alignment and volume confirmation. Uses 4h Donchian channels to establish trend bias, 1h RSI for oversold/overbought entries, and volume spike confirmation to filter false signals. Designed to work in both bull and bear regimes by trading mean reversions within the dominant 4h trend. Targets 15-35 trades/year on 1h timeframe with session filter (08-20 UTC) to reduce noise.

name = "1h_DonchianTrend_RSI_MeanRev_v1"
timeframe = "1h"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for HTF trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1h RSI (14-period) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback period
        # Skip if not in trading session or missing data
        if not in_session[i] or \
           np.isnan(donchian_high_aligned[i]) or \
           np.isnan(donchian_low_aligned[i]) or \
           np.isnan(rsi[i]) or \
           np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price near 4h Donchian lower band + 1h RSI oversold + volume spike
            if (close[i] <= donchian_low_aligned[i] * 1.005 and  # within 0.5% of lower band
                rsi[i] < 30 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price near 4h Donchian upper band + 1h RSI overbought + volume spike
            elif (close[i] >= donchian_high_aligned[i] * 0.995 and  # within 0.5% of upper band
                  rsi[i] > 70 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches 4h Donchian middle OR RSI returns to neutral
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if (close[i] >= donchian_mid or 
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reaches 4h Donchian middle OR RSI returns to neutral
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if (close[i] <= donchian_mid or 
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals