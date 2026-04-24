#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume MA(30).
- Entry: Long when price breaks above Donchian upper(20) AND 1d EMA34 rising AND volume > 1.5 * 12h volume MA(30);
         Short when price breaks below Donchian lower(20) AND 1d EMA34 falling AND volume > 1.5 * 12h volume MA(30).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when EMA34 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian breakouts capture strong momentum moves; 1d EMA34 filter ensures we trade with the daily trend to avoid counter-trend whipsaws; volume confirmation avoids low-conviction breakouts.
- Works in bull markets (buy upper breakouts in uptrend) and bear markets (sell lower breakdowns in downtrend) with trend filter to avoid range-bound false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian(20) breakout frequency with filters.
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
    
    # Get 1d data for EMA34 trend filter and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_slope[0] = 0
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    
    # Get 12h data for Donchian channels (using prior bar extremes)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian channels for each 12h bar based on prior 20 bars
    donchian_high = np.full_like(high_12h, np.nan)
    donchian_low = np.full_like(low_12h, np.nan)
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    # Forward fill for bars < 20
    last_valid_high = donchian_high[20] if not np.isnan(donchian_high[20]) else high_12h[0]
    last_valid_low = donchian_low[20] if not np.isnan(donchian_low[20]) else low_12h[0]
    for i in range(20):
        donchian_high[i] = last_valid_high
        donchian_low[i] = last_valid_low
    
    vol_ma_12h = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    
    # Align all indicators to primary 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 35  # Need sufficient data for EMA34 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (EMA34 slope changes sign)
        if position != 0:
            if position == 1 and ema_34_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_34_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Donchian breakout
        bullish_breakout = curr_high > donchian_high_aligned[i]  # Break above upper band
        bearish_breakout = curr_low < donchian_low_aligned[i]    # Break below lower band
        
        # Trend filter: only trade in direction of 1d EMA34 slope
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        # Volume confirmation (1.5x average volume on both timeframes)
        vol_confirm_12h = curr_volume > 1.5 * vol_ma_12h_aligned[i]
        vol_confirm_1d = curr_volume > 1.5 * vol_ma_1d_aligned[i]
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Donchian upper AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Donchian lower AND downtrend
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

name = "12h_Donchian20_EMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0