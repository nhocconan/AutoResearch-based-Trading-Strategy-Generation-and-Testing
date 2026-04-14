#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# Uses institutional pivot levels with trend strength filter to avoid false breakouts
# Works in bull/bear by only taking breakouts in direction of 1d trend (ADX > 25)
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (ADX) and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for trend strength filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    tr14 = wilder_smoothing(tr, 14)
    plus_dm14 = wilder_smoothing(plus_dm, 14)
    minus_dm14 = wilder_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = wilder_smoothing(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels from previous 1d candle
    # Camarilla formulas based on previous day's range
    prev_close_1d_shift = np.roll(close_1d, 1)
    prev_close_1d_shift[0] = close_1d[0]
    prev_high_1d = np.roll(high_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d = np.roll(low_1d, 1)
    prev_low_1d[0] = low_1d[0]
    
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels
    camarilla_h4 = prev_close_1d_shift + 1.1 * range_1d / 2
    camarilla_l4 = prev_close_1d_shift - 1.1 * range_1d / 2
    camarilla_h3 = prev_close_1d_shift + 1.1 * range_1d / 4
    camarilla_l3 = prev_close_1d_shift - 1.1 * range_1d / 4
    camarilla_h2 = prev_close_1d_shift + 1.1 * range_1d / 6
    camarilla_l2 = prev_close_1d_shift - 1.1 * range_1d / 6
    camarilla_h1 = prev_close_1d_shift + 1.1 * range_1d / 12
    camarilla_l1 = prev_close_1d_shift - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.3x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for ADX calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla H3 with volume filter AND strong trend (ADX > 25)
            if (price > camarilla_h3_aligned[i] and 
                vol > 1.3 * avg_vol[i] and 
                adx_val > 25):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Camarilla L3 with volume filter AND strong trend (ADX > 25)
            elif (price < camarilla_l3_aligned[i] and 
                  vol > 1.3 * avg_vol[i] and 
                  adx_val > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Camarilla L3 OR trend weakens (ADX < 20)
            if price < camarilla_l3_aligned[i] or adx_val < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Camarilla H3 OR trend weakens (ADX < 20)
            if price > camarilla_h3_aligned[i] or adx_val < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_ADX_Volume"
timeframe = "4h"
leverage = 1.0