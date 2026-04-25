#!/usr/bin/env python3
"""
1h Williams Fractal Breakout + 4h EMA50 Trend + Volume Spike
Hypothesis: Williams fractals on 4h timeframe identify significant swing highs/lows that act as key support/resistance.
Breakouts above bearish fractals (sell signals) or below bullish fractals (buy signals) with volume confirmation
and aligned with 4h EMA50 trend capture momentum moves. Designed for 1h timeframe with tight entry conditions
to achieve 15-37 trades/year. Works in bull (breakouts above fractals in uptrend) and bear
(breakouts below fractals in downtrend). Fractals require 2-bar confirmation delay to avoid look-ahead.
Session filter (08-20 UTC) reduces noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for fractals and EMA (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Williams fractals on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_4h, low_4h)
    
    # Align fractals to 1h timeframe with 2-bar confirmation delay
    # Bearish fractal: needs 2 extra 4h bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bearish_fractal, additional_delay_bars=2)
    # Bullish fractal: needs 2 extra 4h bars after center bar for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bullish_fractal, additional_delay_bars=2)
    
    # Calculate EMA50 on 4h close for trend
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (pre-compute hours for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        bearish_fractal_level = bearish_fractal_aligned[i]
        bullish_fractal_level = bullish_fractal_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above bullish fractal AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > bullish_fractal_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below bearish fractal AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < bearish_fractal_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below bullish fractal OR price crosses below EMA (trend change)
            if (curr_low < bullish_fractal_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above bearish fractal OR price crosses above EMA (trend change)
            if (curr_high > bearish_fractal_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_WilliamsFractal_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0