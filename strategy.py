#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX regime filter + volume confirmation
# In trending regime (1d ADX > 25): trade Donchian breakouts in direction of trend
# In ranging regime (1d ADX <= 25): fade Donchian breakouts (mean reversion)
# Uses 1d ADX for regime detection, 4h for entry/exit timing
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: adapts to regime via ADX filter

name = "4h_1d_donchian_adx_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full(len(data), np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Calculate smoothed values
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate DI and DX
    plus_di_14 = np.full(len(df_1d), np.nan)
    minus_di_14 = np.full(len(df_1d), np.nan)
    dx_14 = np.full(len(df_1d), np.nan)
    
    for i in range(14, len(df_1d)):
        if tr_14[i] != 0:
            plus_di_14[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / tr_14[i]) * 100
            if (plus_di_14[i] + minus_di_14[i]) != 0:
                dx_14[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_14_4h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    def rolling_mean(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    volume_ma = rolling_mean(volume, 20)
    volume_ratio = np.divide(volume, volume_ma, out=np.full_like(volume, np.nan), where=volume_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_14_4h[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_14_4h[i]
        vol_ratio = volume_ratio[i]
        
        # Volume confirmation: require volume > 1.5x average
        volume_confirmed = vol_ratio > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when price breaks below Donchian low
                if close[i] <= donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime
                # Exit when price returns to mean (midpoint of Donchian channel)
                donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] >= donchian_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when price breaks above Donchian high
                if close[i] >= donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime
                # Exit when price returns to mean (midpoint of Donchian channel)
                donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] <= donchian_mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime and volume confirmation
            if volume_confirmed:
                if adx > 25:  # Trending regime - trade breakouts in trend direction
                    # Determine 1d trend direction using ADX slope (simplified: use price vs EMA)
                    # We'll use price position relative to Donchian midpoint as proxy
                    donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
                    if close[i] > donchian_mid:
                        # Upward bias - go long on breakout above Donchian high
                        if high[i] > donchian_high[i]:
                            position = 1
                            signals[i] = 0.25
                    else:
                        # Downward bias - go short on breakdown below Donchian low
                        if low[i] < donchian_low[i]:
                            position = -1
                            signals[i] = -0.25
                else:  # Ranging regime - fade breakouts (mean reversion)
                    # Go short when price breaks above Donchian high (expect reversion to mean)
                    if high[i] > donchian_high[i]:
                        position = -1
                        signals[i] = -0.25
                    # Go long when price breaks below Donchian low (expect reversion to mean)
                    elif low[i] < donchian_low[i]:
                        position = 1
                        signals[i] = 0.25
    
    return signals