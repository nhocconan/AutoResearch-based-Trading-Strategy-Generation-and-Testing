#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams Fractal breakout with 4h EMA34 trend filter and volume confirmation
# Williams Fractals on 1h timeframe provide intraday support/resistance levels effective in both bull and bear markets
# 4h EMA34 ensures trades only with intermediate-term trend, reducing false breakouts in choppy markets
# Volume confirmation at 1.5x average filters low-participation moves
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Discrete sizing 0.20 to minimize fee churn while maintaining adequate position size
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe

name = "1h_WilliamsFractal_Breakout_4hEMA34_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Fractals on 1h (requires 5-bar window: n-2, n-1, n, n+1, n+2)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Bearish fractal: current high is highest of [n-2, n-1, n, n+1, n+2]
    bearish_fractal = (high_series.rolling(window=5, center=True, min_periods=5).max() == high_series).values
    # Bullish fractal: current low is lowest of [n-2, n-1, n, n+1, n+2]
    bullish_fractal = (low_series.rolling(window=5, center=True, min_periods=5).min() == low_series).values
    
    # 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
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
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            i >= len(bearish_fractal) or i >= len(bullish_fractal)):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish fractal confirmed AND price > 4h EMA34 AND volume spike
            if (bullish_fractal[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: Bearish fractal confirmed AND price < 4h EMA34 AND volume spike
            elif (bearish_fractal[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below 4h EMA34 OR bearish fractal forms
            if close[i] < ema_34_4h_aligned[i] or bearish_fractal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price rises above 4h EMA34 OR bullish fractal forms
            if close[i] > ema_34_4h_aligned[i] or bullish_fractal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals