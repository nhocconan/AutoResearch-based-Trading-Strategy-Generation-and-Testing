#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend filter.
- Entry: Long when price breaks above Donchian(20) high AND 12h EMA50 > 12h EMA50(previous) (uptrend) AND volume > 1.5 * 6h volume MA(20);
         Short when price breaks below Donchian(20) low AND 12h EMA50 < 12h EMA50(previous) (downtrend) AND volume > 1.5 * 6h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when EMA50 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide structure-based breakouts; EMA50 trend filter ensures we trade with the intermediate trend; volume confirmation (1.5x) avoids false breakouts.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_12h - np.roll(ema_50_12h, 1)
    ema_50_slope[0] = 0
    
    # Calculate Donchian(20) channels on primary 6h data
    # Upper = rolling max(high, 20), Lower = rolling min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_50_slope)
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)  # same timeframe
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)  # same timeframe
    vol_ma_6h_aligned = align_htf_to_ltf(prices, prices, vol_ma_6h)  # same timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
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
        bullish_breakout = curr_high > donchian_upper_aligned[i]  # Break above upper band
        bearish_breakout = curr_low < donchian_lower_aligned[i]    # Break below lower band
        
        # Trend filter: only trade in direction of 12h EMA50 slope
        uptrend = ema_50_slope_aligned[i] > 0
        downtrend = ema_50_slope_aligned[i] < 0
        
        # Volume confirmation (1.5x average volume)
        vol_confirm = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
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

name = "6h_Donchian20_EMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0