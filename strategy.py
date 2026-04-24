#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and Donchian channels (for structure).
- Entry: Long when price breaks above 1d Donchian upper(20) AND 1d EMA34 > 1d EMA34(previous) (uptrend) AND volume > 2.0 * 12h volume MA(50);
         Short when price breaks below 1d Donchian lower(20) AND 1d EMA34 < 1d EMA34(previous) (downtrend) AND volume > 2.0 * 12h volume MA(50).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when 1d EMA34 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Uses higher timeframe structure (1d) for breakout levels and trend filter, reducing false breakouts and whipsaws.
- Volume confirmation on 12h timeframe ensures institutional participation.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on 1d Donchian(20) breakout frequency with filters.
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
    
    # Get 1d data for Donchian channels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 and its slope
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_slope[0] = 0
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to primary 12h timeframe
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 100  # Need sufficient data for EMA34 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (1d EMA34 slope changes sign)
        if position != 0:
            if position == 1 and ema_34_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_34_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and 1d Donchian breakout
        bullish_breakout = curr_high > donchian_high_1d_aligned[i]  # Break above upper channel
        bearish_breakout = curr_low < donchian_low_1d_aligned[i]    # Break below lower channel
        
        # Trend filter: only trade in direction of 1d EMA34 slope
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        # Volume confirmation (require strong volume on 12h timeframe)
        vol_confirm = curr_volume > 2.0 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above 1d Donchian upper AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below 1d Donchian lower AND downtrend
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