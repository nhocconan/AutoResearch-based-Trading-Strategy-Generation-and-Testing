#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend filter and 1d for Donchian channel calculation.
- Entry: Long when price breaks above 1d Donchian upper channel AND 4h EMA50 > 4h EMA50(previous) (uptrend) AND volume > 1.5 * 1h volume MA(20);
         Short when price breaks below 1d Donchian lower channel AND 4h EMA50 < 4h EMA50(previous) (downtrend) AND volume > 1.5 * 1h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when 4h EMA50 slope changes sign).
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide volatility-based structure; EMA50 trend filter ensures we trade with the intermediate trend; volume confirmation avoids false breakouts.
- Works in bull markets (buy upper breakouts in uptrend) and bear markets (sell lower breakouts in downtrend) with trend filter to avoid counter-trend whipsaws.
- Session filter (08-20 UTC) to reduce noise trades during low-liquidity periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels for today based on previous 20 days
    # Upper = max(high_1d[-20:-1]), Lower = min(low_1d[-20:-1])
    # We use rolling window on 1d data, then align
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_4h - np.roll(ema_50_4h, 1)
    ema_50_slope[0] = 0
    
    # Get 1h data for volume MA
    volume_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_50_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08-20 UTC (inclusive)
    # prices.index is DatetimeIndex, .hour works directly
    hours = prices.index.hour
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (4h EMA50 slope changes sign)
        if position != 0:
            if position == 1 and ema_50_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_50_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Donchian breakout
        bullish_breakout = curr_high > donchian_upper_aligned[i]  # Break above upper channel
        bearish_breakout = curr_low < donchian_lower_aligned[i]    # Break below lower channel
        
        # Trend filter: only trade in direction of 4h EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * volume_ma_1h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Donchian upper AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.20
                    position = 1
                # Short: Price breaks below Donchian lower AND downtrend
                elif bearish_breakout and downtrend:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_EMA50_Trend_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0