#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike
# Williams fractals identify key swing highs/lows that act as support/resistance.
# Breakout above bearish fractal (sell fractal) with 1d EMA34 uptrend and volume spike = long.
# Breakout below bullish fractal (buy fractal) with 1d EMA34 downtrend and volume spike = short.
# Uses completed fractals only (2-bar delay) to avoid look-ahead.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing.

name = "6h_WilliamsFractal_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter and Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams fractals on 1d data
    from mtf_data import compute_williams_fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Bearish fractal needs 2 extra 1d bars for confirmation (already handled by compute_williams_fractals?)
    # According to Rule 2b: Williams fractal needs 2 extra 1d bars after the center bar
    # So we apply additional_delay_bars=2 when aligning
    
    # Align 1d EMA34 to 6h timeframe (needs only completed 1d candle)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align Williams fractals to 6h timeframe with extra 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: 20-period EMA on 6h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Uptrend: price above 1d EMA34
        uptrend = close[i] > ema_34_aligned[i]
        # Downtrend: price below 1d EMA34
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal (sell fractal) in uptrend with volume spike
            if close[i] > bearish_fractal_aligned[i] and uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (buy fractal) in downtrend with volume spike
            elif close[i] < bullish_fractal_aligned[i] and downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below bullish fractal or loses uptrend
            if close[i] < bullish_fractal_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above bearish fractal or loses downtrend
            if close[i] > bearish_fractal_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals