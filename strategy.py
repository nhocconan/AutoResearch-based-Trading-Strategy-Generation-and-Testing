#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h/1d HTF for signal direction and 1h for precise entry timing.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h Donchian(20) breakout + 1d EMA50 trend filter for direction.
- Entry: Long when price breaks above 1h Donchian upper(10) AND 4h Donchian trend is up AND 1d EMA50 rising AND volume > 1.5 * 1h volume MA(20);
         Short when price breaks below 1h Donchian lower(10) AND 4h Donchian trend is down AND 1d EMA50 falling AND volume > 1.5 * 1h volume MA(20).
- Exit: Close-based reversal or trend change (signal=0 when 1d EMA50 slope changes sign).
- Signal size: 0.20 discrete to minimize fee drag.
- Uses 1h for timing precision but relies on 4h/1d for structure to avoid overtrading.
- Session filter: 08-20 UTC to avoid low-liquidity periods.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) with dual timeframe trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope_1d = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_slope_1d[0] = 0
    
    # Get 4h data for Donchian trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_trend_4h = donchian_high_4h - donchian_low_4h  # widening = trending
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2
    donchian_slope_4h = donchian_mid_4h - np.roll(donchian_mid_4h, 1)
    donchian_slope_4h[0] = 0
    
    # Get 1h data for entry timing
    high_1h = high
    low_1h = low
    volume_1h = volume
    
    # Calculate 1h Donchian channels (10-period for tighter entries)
    donchian_high_1h = pd.Series(high_1h).rolling(window=10, min_periods=10).max().values
    donchian_low_1h = pd.Series(low_1h).rolling(window=10, min_periods=10).min().values
    vol_ma_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope_1d)
    donchian_slope_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_slope_4h)
    donchian_high_1h_aligned = align_htf_to_ltf(prices, prices, donchian_high_1h)  # same timeframe
    donchian_low_1h_aligned = align_htf_to_ltf(prices, prices, donchian_low_1h)
    vol_ma_1h_aligned = align_htf_to_ltf(prices, prices, vol_ma_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 100  # Need sufficient data for all calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_slope_1d_aligned[i]) or 
            np.isnan(donchian_slope_4h_aligned[i]) or np.isnan(donchian_high_1h_aligned[i]) or 
            np.isnan(donchian_low_1h_aligned[i]) or np.isnan(vol_ma_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = prices.index[i].hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change on 1d EMA50
        if position != 0:
            if position == 1 and ema_50_slope_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_50_slope_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        bullish_breakout = curr_high > donchian_high_1h_aligned[i]
        bearish_breakout = curr_low < donchian_low_1h_aligned[i]
        
        # HTF trend filters: 4h Donchian slope + 1d EMA50 slope
        uptrend_4h = donchian_slope_4h_aligned[i] > 0
        downtrend_4h = donchian_slope_4h_aligned[i] < 0
        uptrend_1d = ema_50_slope_1d_aligned[i] > 0
        downtrend_1d = ema_50_slope_1d_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_1h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above 1h Donchian upper AND both timeframes show uptrend
                if bullish_breakout and uptrend_4h and uptrend_1d:
                    signals[i] = 0.20
                    position = 1
                # Short: Price breaks below 1h Donchian lower AND both timeframes show downtrend
                elif bearish_breakout and downtrend_4h and downtrend_1d:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Donchian10_4hTrend_1dEMA50_Volume_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0