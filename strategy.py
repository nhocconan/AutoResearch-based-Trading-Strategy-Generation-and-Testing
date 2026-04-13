#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w ADX regime filter
    # Long: price > Donchian high(20) AND volume > 1.5x 20-period avg AND 1w ADX > 25
    # Short: price < Donchian low(20) AND volume > 1.5x 20-period avg AND 1w ADX > 25
    # Exit: price crosses Donchian midpoint OR volume dry-up
    # Using 12h timeframe for low trade frequency, Donchian for structure,
    # volume for confirmation, 1w ADX for trend regime filter (avoid chop).
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily volume MA(20) for confirmation
    vol_ma = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma[i] = np.mean(df_1d['volume'].values[i-20:i])
    volume_spike_1d = df_1d['volume'].values > (1.5 * vol_ma)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate weekly ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[1:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dpi = wilders_smoothing(dm_plus, 14)
    dmi = wilders_smoothing(dm_minus, 14)
    
    # Avoid division by zero
    di_plus = 100 * dpi / atr
    di_minus = 100 * dmi / atr
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donch_high[i] = np.max(high[i-lookback:i])
        donch_low[i] = np.min(low[i-lookback:i])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market
        trending_regime = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume_spike_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume confirmation + trend regime
        long_entry = (close[i] > donch_high[i]) and vol_confirm and trending_regime
        short_entry = (close[i] < donch_low[i]) and vol_confirm and trending_regime
        
        # Exit logic: price crosses midpoint OR volume dry-up
        long_exit = (close[i] < donch_mid[i]) or not vol_confirm
        short_exit = (close[i] > donch_mid[i]) or not vol_confirm
        
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

name = "12h_1d_1w_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0