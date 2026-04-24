#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and Williams Fractal calculation.
- Entry: Long when price breaks above recent bullish fractal AND 1d EMA34 > 1d EMA34(previous) (uptrend) AND volume > 1.5 * 12h volume MA(50);
         Short when price breaks below recent bearish fractal AND 1d EMA34 < 1d EMA34(previous) (downtrend) AND volume > 1.5 * 12h volume MA(50).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when EMA34 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Fractals identify potential reversal points; EMA34 trend filter ensures we trade with the higher timeframe trend; volume confirmation avoids false breakouts.
- Works in bull markets (buy fractal breakouts in uptrend) and bear markets (sell fractal breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~80 total over 4 years (~20/year) based on Williams Fractal frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_slope[0] = 0
    
    # Calculate Williams Fractals on 1d data (need 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Align HTF indicators to primary 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=50, min_periods=50).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for EMA34 and fractals
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
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
        
        # Entry conditions with volume confirmation and fractal breakout
        bullish_breakout = curr_high > bullish_fractal_aligned[i]  # Break above bullish fractal
        bearish_breakout = curr_low < bearish_fractal_aligned[i]    # Break below bearish fractal
        
        # Trend filter: only trade in direction of 1d EMA34 slope
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        # Volume confirmation (1.5x average volume)
        vol_confirm = curr_volume > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above bullish fractal AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below bearish fractal AND downtrend
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

name = "12h_WilliamsFractal_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0