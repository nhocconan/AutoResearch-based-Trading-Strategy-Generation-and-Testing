#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d ATR-based volume spike and chop regime filter.
    # Long when price breaks above Donchian(20) high with volume spike and chop < 61.8 (trending).
    # Short when price breaks below Donchian(20) low with volume spike and chop < 61.8.
    # Exit on opposite Donchian(10) break to reduce whipsaw. Uses discrete size 0.25.
    # Target: 75-200 trades over 4 years (19-50/year) to avoid fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR (14-period) for volatility-based volume spike threshold
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(high)
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d volume mean (20-period) with min_periods
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (14-period) for regime filter
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Wilder's smoothing for ATR
        if len(tr) > period:
            atr[period] = np.nansum(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Sum of ATR over period
        sum_atr = np.zeros_like(close)
        for i in range(period, len(close)):
            sum_atr[i] = np.nansum(atr[i-period+1:i+1])
        # Max high - min low over period
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(period-1, len(high)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        # Chop = 100 * log10(sum(ATR) / (maxH - minL)) / log10(period)
        range_hl = max_high - min_low
        chop = np.full_like(close, 50.0)  # default to neutral
        for i in range(period, len(close)):
            if range_hl[i] > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / range_hl[i]) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Calculate 4h Donchian channels (20-period for entry, 10-period for exit)
    def donchian_channel(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_high_20, donchian_low_20 = donchian_channel(high, low, 20)
    donchian_high_10, donchian_low_10 = donchian_channel(high, low, 10)
    
    # Align HTF indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2.0 * 20-period mean (volume spike)
        volume_confirmation = volume_1d_aligned[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Regime filter: chop < 61.8 indicates trending market (avoid choppy ranging)
        regime_filter = chop_aligned[i] < 61.8
        
        # Entry conditions: price breaks Donchian(20) with volume confirmation and trend regime
        long_entry = (close[i] > donchian_high_20[i] and volume_confirmation and regime_filter)
        short_entry = (close[i] < donchian_low_20[i] and volume_confirmation and regime_filter)
        
        # Exit conditions: price breaks opposite Donchian(10) to reduce whipsaw
        long_exit = close[i] < donchian_low_10[i]
        short_exit = close[i] > donchian_high_10[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0