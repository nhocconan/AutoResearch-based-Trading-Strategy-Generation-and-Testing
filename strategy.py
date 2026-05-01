#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA50 trend filter and volume spike confirmation
# Williams Fractals identify potential reversal points: bullish fractal = low with two higher lows on each side,
# bearish fractal = high with two lower highs on each side. Breakouts from fractal levels with volume confirmation
# and 1d trend filter capture institutional participation in trending moves. Designed for low trade frequency:
# ~10-20 trades/year per symbol with 0.25 sizing. Works in bull/bear markets by following 1d EMA50 trend.

name = "6h_WilliamsFractal_Breakout_1dEMA50_Trend_Volume_v1"
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
    
    # 1d HTF data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Fractals (5-bar: 2 left, center, 2 right)
    # Bullish fractal: low[i] is lowest among low[i-2:i+3]
    # Bearish fractal: high[i] is highest among high[i-2:i+3]
    bullish_fractal = np.zeros(n, dtype=bool)
    bearish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if low[i] == min(low[i-2:i+3]):
            bullish_fractal[i] = True
        if high[i] == max(high[i-2:i+3]):
            bearish_fractal[i] = True
    
    # Fractal levels (only valid at fractal points)
    bullish_level = np.full(n, np.nan)
    bearish_level = np.full(n, np.nan)
    bullish_level[bullish_fractal] = low[bullish_fractal]
    bearish_level[bearish_fractal] = high[bearish_fractal]
    
    # Volume confirmation: volume > 1.8 * 30-period EMA (higher threshold for fewer trades)
    vol_series = pd.Series(volume)
    vol_ema_30 = vol_series.ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ema_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(60, 30, 4)  # Need 1d EMA50, volume EMA30, fractal lookback
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_30[i]) or 
            np.isnan(bullish_level[i]) and np.isnan(bearish_level[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend:
                # Long: break above bearish fractal resistance with volume spike
                if not np.isnan(bearish_level[i]) and close[i] > bearish_level[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend:
                # Short: break below bullish fractal support with volume spike
                if not np.isnan(bullish_level[i]) and close[i] < bullish_level[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid sideways markets
        
        elif position == 1:  # Long position
            # Exit: price closes below bullish fractal support (failed support)
            if not np.isnan(bullish_level[i]) and close[i] < bullish_level[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above bearish fractal resistance (failed resistance)
            if not np.isnan(bearish_level[i]) and close[i] > bearish_level[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals