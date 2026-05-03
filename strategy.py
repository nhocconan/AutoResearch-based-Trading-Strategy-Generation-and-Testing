#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation
# Williams Fractals identify key swing points where price reverses. A breakout above the most
# recent bearish fractal (or below bullish fractal) with volume confirmation and aligned
# weekly trend captures strong momentum moves. Weekly EMA50 filter ensures we trade with
# the higher timeframe trend, reducing false breakouts in ranging markets. Designed for
# 12-37 trades/year on 6h to minimize fee drag while maintaining edge in bull/bear markets.

name = "6h_WilliamsFractal_Breakout_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w, additional_delay_bars=0)
    
    # Calculate Williams Fractals on 1d (requires 2-bar confirmation after center)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Initialize fractal arrays
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    # Williams Fractal: bearish = high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # bullish = low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align fractals to 6h timeframe with 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for EMA50
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 2x 20-period EMA of volume
        vol_lookback = min(20, i+1)
        vol_slice = volume[max(0, i-19):i+1]
        if len(vol_slice) > 0:
            vol_ema_20 = pd.Series(vol_slice).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        # Most recent completed fractal values (using aligned arrays)
        recent_bearish = bearish_fractal_aligned[i]
        recent_bullish = bullish_fractal_aligned[i]
        
        # Breakout conditions
        breakout_long = (not np.isnan(recent_bearish)) and (close[i] > recent_bearish) and volume_spike
        breakout_short = (not np.isnan(recent_bullish)) and (close[i] < recent_bullish) and volume_spike
        
        if position == 0:
            # Long: break above bearish fractal in weekly uptrend with volume spike
            if breakout_long and ema_50_1w_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below bullish fractal in weekly downtrend with volume spike
            elif breakout_short and ema_50_1w_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly EMA50 or breaks below bullish fractal
            if close[i] < ema_50_1w_aligned[i] or (not np.isnan(recent_bullish) and close[i] < recent_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly EMA50 or breaks above bearish fractal
            if close[i] > ema_50_1w_aligned[i] or (not np.isnan(recent_bearish) and close[i] > recent_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals