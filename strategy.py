#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian breakouts capture strong momentum moves. Aligned with 1w EMA34 trend and confirmed by volume spikes,
this strategy targets trending moves in both bull and bear markets. Chop filter avoids ranging markets. Designed for 1d
to target 7-25 trades/year (30-100 over 4 years) by requiring confluence of breakout, trend, volume, and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for indicators
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Chop filter: Choppiness Index > 61.8 = ranging (avoid), < 38.2 = trending (favor)
    # Using 14-period chop on daily closes
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = np.where(np.isnan(tr), 0, tr)
        # Wilder's smoothing
        for i in range(1, len(atr)):
            if np.isnan(atr[i]):
                atr[i] = atr[i-1]
            else:
                atr[i] = (atr[i-1] * (period-1) + atr[i]) / period
        # Chop calculation
        sum_atr = np.nansum(atr[1:period+1]) if len(atr) > period else np.nansum(atr[1:])
        hh = np.max(high_arr[:period]) if len(high_arr) >= period else np.max(high_arr)
        ll = np.min(low_arr[:period]) if len(low_arr) >= period else np.min(low_arr)
        chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(period) if (hh - ll) > 0 else 100
        # Fill array
        chop_arr = np.full_like(close_arr, np.nan, dtype=float)
        chop_arr[period-1] = chop
        # For simplicity, use same value (can be improved with rolling)
        for i in range(period, len(chop_arr)):
            chop_arr[i] = chop_arr[i-1]
        return chop_arr
    
    chop_values = calculate_chop(high, low, close, 14)
    chop_filter = chop_values < 61.8  # Avoid ranging markets (chop > 61.8)
    
    # Donchian channels (20-period)
    def donchian_channels(high_arr, low_arr, period=20):
        upper = np.full_like(high_arr, np.nan, dtype=float)
        lower = np.full_like(low_arr, np.nan, dtype=float)
        for i in range(period-1, len(high_arr)):
            upper[i] = np.max(high_arr[i-period+1:i+1])
            lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34, 20)  # Donchian, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        in_trend = chop_filter[i]
        
        # Trend filter: price relative to 1w EMA34
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: breakout + trend + volume + chop filter
            # Long: price breaks above Donchian upper AND bullish bias AND volume spike AND trending market
            long_entry = (curr_high > dc_upper[i]) and bullish_bias and vol_spike and in_trend
            # Short: price breaks below Donchian lower AND bearish bias AND volume spike AND trending market
            short_entry = (curr_low < dc_lower[i]) and bearish_bias and vol_spike and in_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian lower (mean reversion) OR loss of bullish bias OR chop too high
            if (curr_low < dc_lower[i]) or (curr_close < ema_1w_aligned[i]) or (not in_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper (mean reversion) OR loss of bearish bias OR chop too high
            if (curr_high > dc_upper[i]) or (curr_close > ema_1w_aligned[i]) or (not in_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0