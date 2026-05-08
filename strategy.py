#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume spike and 1d RSI filter
# Donchian breakouts capture momentum in trending markets. Volume spike confirms
# institutional participation. RSI > 50 for longs and < 50 for shorts ensures
# we trade with momentum, not against it. This combination works in both bull
# and bear markets by filtering for strong momentum with volume confirmation.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

name = "6h_Donchian20_1dVolume_1dRSI"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 6h data
    # We need at least 20 periods to calculate
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume spike detection on 1d
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_spike = df_1d['volume'].values > (vol_ma.values * 2.0)
    vol_spike_6h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # RSI(14) on 1d
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_buy = rsi > 50
    rsi_sell = rsi < 50
    rsi_buy_6h = align_htf_to_ltf(prices, df_1d, rsi_buy)
    rsi_sell_6h = align_htf_to_ltf(prices, df_1d, rsi_sell)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_spike_6h[i]) or np.isnan(rsi_buy_6h[i]) or 
            np.isnan(rsi_sell_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume spike, RSI > 50
            if close[i] > highest_high[i] and vol_spike_6h[i] and rsi_buy_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume spike, RSI < 50
            elif close[i] < lowest_low[i] and vol_spike_6h[i] and rsi_sell_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or RSI < 50
            if close[i] < lowest_low[i] or not rsi_buy_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or RSI > 50
            if close[i] > highest_high[i] or not rsi_sell_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals