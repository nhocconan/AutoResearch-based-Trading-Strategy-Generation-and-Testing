#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter.
- Entry: Long when price breaks above 20-bar Donchian high AND 1d EMA50 > 1d EMA50(previous) (uptrend) AND volume > 1.5 * 12h volume MA(20);
         Short when price breaks below 20-bar Donchian low AND 1d EMA50 < 1d EMA50(previous) (downtrend) AND volume > 1.5 * 12h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when EMA50 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide robust structure; EMA50 trend filter ensures we trade with the intermediate trend; volume confirmation avoids false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
- Focus on BTC/ETH as primary targets, SOL as secondary.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    ema_50_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_slope[0] = 0
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels on 12h timeframe
    # Upper band = 20-period high, Lower band = 20-period low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to primary 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (EMA50 slope changes sign)
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
        bullish_breakout = curr_high > donchian_high_aligned[i]  # Break above upper band
        bearish_breakout = curr_low < donchian_low_aligned[i]    # Break below lower band
        
        # Trend filter: only trade in direction of 1d EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Donchian upper band AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Donchian lower band AND downtrend
                elif bearish_breakout and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0