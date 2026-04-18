#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w ADX trend filter.
# Long when price breaks above Donchian high(20) with volume > 1.5x 20-period average and 1w ADX > 25.
# Short when price breaks below Donchian low(20) with volume > 1.5x 20-period average and 1w ADX > 25.
# Exit when price returns to Donchian midpoint.
# Uses Donchian channels for structure, volume surge for conviction, ADX for trend strength filter.
# Designed for ~10-25 trades/year per symbol on daily timeframe.
name = "1d_Donchian_20_Volume_ADX_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w timeframe
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = wilder_smooth(tr, 14)
    plus_di_1w = 100 * wilder_smooth(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilder_smooth(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = wilder_smooth(dx_1w, 14)
    
    # Handle division by zero and NaN
    adx_1w = np.where((plus_di_1w + minus_di_1w) == 0, 0, adx_1w)
    adx_1w = np.nan_to_num(adx_1w, nan=0.0)
    
    # Align ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Donchian Channel (20) on 1d
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        adx_val = adx_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume surge and strong trend (ADX > 25)
            if high_val > upper and vol_filter and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume surge and strong trend (ADX > 25)
            elif low_val < lower and vol_filter and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian midpoint
            if close_val <= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian midpoint
            if close_val >= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals