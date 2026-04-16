#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
# Uses Donchian(20) on 12h timeframe for breakout signals, filtered by 1d EMA(50) trend direction
# and confirmed by volume > 1.8x average. Only trades in direction of higher timeframe trend.
# Works in both bull and bear markets by following the 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 12h Donchian(20) channels ===
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # === 1d EMA(50) for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h volume spike detection ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.8 * vol_ma_20_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low or trend reverses
            if price < donch_low or ema_trend < close_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high or trend reverses
            if price > donch_high or ema_trend > close_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike and price near Donchian levels
            if vol_spike_val:
                # Go long when price breaks above Donchian high in uptrend
                if price > donch_high and close_12h[i] > ema_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below Donchian low in downtrend
                elif price < donch_low and close_12h[i] < ema_trend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_DonchianBreakout_1dEMA50_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0