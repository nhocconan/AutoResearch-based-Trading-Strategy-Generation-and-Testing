#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h ADX trend filter and volume confirmation.
# ADX > 25 indicates strong trend; only trade breakouts in trending markets.
# In non-trending markets (ADX < 20), remain flat to avoid whipsaw.
# Volume > 1.5x 20-period average confirms breakout strength.
# Fixed position size of 0.25 to limit risk and reduce trade frequency.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "4h_Donchian_ADX_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period)
    def true_range(high, low, close):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        return tr
    
    def directional_movement(high, low):
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        return plus_dm, minus_dm
    
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr = true_range(high_12h, low_12h, close_12h)
    plus_dm, minus_dm = directional_movement(high_12h, low_12h)
    
    atr_12h = wilders_smooth(tr, 14)
    plus_di_12h = 100 * wilders_smooth(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smooth(minus_dm, 14) / atr_12h
    dx = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smooth(dx, 14)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high_1d, 20)
    donch_low = rolling_min(low_1d, 20)
    
    # Align indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # ADX (14*2+6), Donchian (20), volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend strength filter: ADX > 25 for strong trend
        is_trending = adx_val > 25
        # Weak trend or ranging: ADX < 20
        is_ranging = adx_val < 20
        
        if position == 0:
            # Only enter in trending markets with volume confirmation
            if is_trending and volume_confirmed:
                if price > donch_high_val:
                    signals[i] = 0.25
                    position = 1
                elif price < donch_low_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit when price crosses below Donchian low or ADX weakens
            if price < donch_low_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above Donchian high or ADX weakens
            if price > donch_high_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals