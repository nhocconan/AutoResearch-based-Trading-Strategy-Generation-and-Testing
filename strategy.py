#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend filter.
- Entry: Long when price breaks above Donchian(20) high AND 1w EMA34 > 1w EMA34(previous) (uptrend) AND volume > 1.5 * 12h volume MA(20);
         Short when price breaks below Donchian(20) low AND 1w EMA34 < 1w EMA34(previous) (downtrend) AND volume > 1.5 * 12h volume MA(20).
- Exit: Trend change (signal=0 when 1w EMA34 slope changes sign).
- Signal size: 0.30 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide structure-based breakouts; EMA34 trend filter ensures we trade with the long-term trend; volume confirmation (1.5x) avoids false breakouts.
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34_1w - np.roll(ema_34_1w, 1)
    ema_34_slope[0] = 0
    
    # Calculate Donchian(20) channels on primary 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_34_slope)
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)  # same timeframe
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)  # same timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, prices, vol_ma_12h)  # same timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (1w EMA34 slope changes sign)
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
        bullish_breakout = curr_high > donchian_upper_aligned[i]  # Break above upper band
        bearish_breakout = curr_low < donchian_lower_aligned[i]    # Break below lower band
        
        # Trend filter: only trade in direction of 1w EMA34 slope
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        # Volume confirmation (1.5x average volume)
        vol_confirm = curr_volume > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Donchian upper AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.30
                    position = 1
                # Short: Price breaks below Donchian lower AND downtrend
                elif bearish_breakout and downtrend:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_EMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0