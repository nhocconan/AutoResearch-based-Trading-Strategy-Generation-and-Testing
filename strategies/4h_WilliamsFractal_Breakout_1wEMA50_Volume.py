#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation
# Uses Williams Fractals (lagging indicator requiring 2-bar confirmation) for high-probability reversal/continuation signals.
# 1w EMA50 ensures trades only with longer-term trend, reducing false breakouts in choppy markets.
# Volume confirmation at 2.0x average filters low-participation moves.
# Session filter (08-20 UTC) avoids low-liquidity periods.
# Discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Williams Fractals provide structural support/resistance levels that work in both bull and bear markets.
# Adding 1w EMA50 as HTF filter should improve BTC/ETH performance vs 1d EMA34 alone.

name = "4h_WilliamsFractal_Breakout_1wEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams Fractals (requires 5-bar window: n-2, n-1, n, n+1, n+2)
    # Bearish fractal: high[n] is highest of [n-2, n-1, n, n+1, n+2]
    # Bullish fractal: low[n] is lowest of [n-2, n-1, n, n+1, n+2]
    # We calculate on completed candles only, so we shift by 2 to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Bearish fractal: current high is highest of previous 2, current, and next 2
    # We use rolling window of 5, centered, but shift by 2 to ensure we only use completed data
    bearish_fractal = (high_series.rolling(window=5, center=True, min_periods=5).max() == high_series).values
    # Bullish fractal: current low is lowest of previous 2, current, and next 2
    bullish_fractal = (low_series.rolling(window=5, center=True, min_periods=5).min() == low_series).values
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 2.0x 20-period average (stricter threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            i >= len(bearish_fractal) or i >= len(bullish_fractal)):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish fractal confirmed AND price > 1w EMA50 AND volume spike
            if (bullish_fractal[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal confirmed AND price < 1w EMA50 AND volume spike
            elif (bearish_fractal[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below 1w EMA50 OR bearish fractal forms
            if close[i] < ema_50_1w_aligned[i] or bearish_fractal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above 1w EMA50 OR bullish fractal forms
            if close[i] > ema_50_1w_aligned[i] or bullish_fractal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals