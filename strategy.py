#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Uses ADX > 25 to filter for trending markets only, reducing false breakouts in ranging markets.
# Entry: Long when price breaks above Donchian high with ADX > 25 and volume > 1.5x average.
# Short when price breaks below Donchian low with ADX > 25 and volume > 1.5x average.
# Exit: When price crosses the Donchian midline or reverses with opposite breakout.
# Position size: 0.25 to limit drawdown during drawdown periods.
# Target: 20-40 trades/year to stay within frequency limits and minimize fee drag.
name = "4h_Donchian20_ADX25_VolumeFilter"
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
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate directional movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing (14-period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Calculate TR and DM arrays
    tr = np.concatenate([[np.nan], true_range(high_1d[1:], low_1d[1:], close_1d[:-1])])
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smooth TR, +DM, -DM
    atr = wilder_smooth(tr, 14)
    plus_di_smooth = wilder_smooth(plus_dm, 14)
    minus_di_smooth = wilder_smooth(minus_dm, 14)
    
    # Calculate DI and DX
    plus_di = 100 * plus_di_smooth / atr
    minus_di = 100 * minus_di_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
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
    
    donch_high = rolling_max(high_4h, 20)
    donch_low = rolling_min(low_4h, 20)
    
    # Align indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure ADX (14*2+6), Donchian (20), and volume MA are ready
    
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
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx_val > 25
        
        if position == 0:
            # Enter only in trending markets with volume confirmation
            if is_trending and volume_confirmed:
                if price > donch_high_val:
                    signals[i] = 0.25
                    position = 1
                elif price < donch_low_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses Donchian midline or reverse breakout
            midline = (donch_high_val + donch_low_val) / 2
            if price < midline or (price < donch_low_val and volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses Donchian midline or reverse breakout
            midline = (donch_high_val + donch_low_val) / 2
            if price > midline or (price > donch_high_val and volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals