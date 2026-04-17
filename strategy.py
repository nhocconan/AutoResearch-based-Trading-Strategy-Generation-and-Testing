# 4h Donchian Breakout with Volume Confirmation and Volatility Filter
# Strategy: Long when price breaks above 4h Donchian upper channel (20-period) with volume confirmation (>1.5x 20-period avg volume)
#           Short when price breaks below 4h Donchian lower channel (20-period) with volume confirmation
#           Exit when price crosses the Donchian midline (10-period average of upper/lower)
#           Uses 1d ATR for volatility filter (only trade when ATR > 50th percentile of 50-period ATR)
#           Designed for 4h timeframe with 20-50 trades/year target
# Works in bull (breakouts continue) and bear (breakdowns continue) markets due to symmetric long/short logic
# Volume confirmation reduces false breakouts
# Volatility filter avoids choppy markets
# Position size: 0.25 (25% of capital)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian Channels (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(low_4h, np.nan)
    for i in range(len(high_4h)):
        if i >= 19:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
        elif i > 0:
            donchian_high[i] = np.max(high_4h[max(0, i-9):i+1])
            donchian_low[i] = np.min(low_4h[max(0, i-9):i+1])
        else:
            donchian_high[i] = high_4h[0]
            donchian_low[i] = low_4h[0]
    
    # Donchian midline (for exit)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === 4h Volume Confirmation ===
    volume_4h = df_4h['volume'].values
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume_4h > vol_ma_20 * 1.5
    
    # === 1d ATR Volatility Filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr = np.zeros_like(high_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Calculate ATR (14-period)
    atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            if i == 0:
                atr_14[i] = tr[i]
            else:
                atr_14[i] = np.mean(tr[:i+1])
        else:
            atr_14[i] = np.mean(tr[i-13:i+1])
    
    # ATR percentile (50-period) for volatility regime
    atr_percentile = np.full_like(atr_14, np.nan)
    for i in range(len(atr_14)):
        if i >= 49:
            window = atr_14[i-49:i+1]
            rank = np.sum(window <= atr_14[i]) / len(window)
            atr_percentile[i] = rank * 100
        elif i > 0:
            window = atr_14[max(0, i-24):i+1]
            rank = np.sum(window <= atr_14[i]) / len(window)
            atr_percentile[i] = rank * 100
        else:
            atr_percentile[i] = 50.0
    
    # Volatility filter: only trade when ATR > 50th percentile (avoid choppy markets)
    vol_filter = atr_percentile >= 50
    
    # === Align indicators to main timeframe ===
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_4h, vol_confirm)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i]) or 
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + volatility filter
            if (close[i] > donchian_high_aligned[i] and 
                vol_confirm_aligned[i] and 
                vol_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low + volume confirmation + volatility filter
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_confirm_aligned[i] and 
                  vol_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0