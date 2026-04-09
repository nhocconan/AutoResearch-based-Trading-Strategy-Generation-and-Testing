#!/usr/bin/env python3
# 12h_williams_fractal_breakout_volume_v1
# Hypothesis: 12h strategy using Williams Fractals for breakout detection, volume confirmation, and 1w EMA trend filter.
# Long when price breaks above latest bearish fractal with volume > 1.5x 20-period average and close > 1w EMA50.
# Short when price breaks below latest bullish fractal with volume > 1.5x 20-period average and close < 1w EMA50.
# Exit when price returns to the opposite fractal level or EMA50 crossover.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: fractals capture key reversal points, volume confirms breakout conviction, 1w EMA filter ensures alignment with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

name = "12h_williams_fractal_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Fractals on 1d timeframe for key levels
    df_1d = get_htf_data(prices, '1d')
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar delay for fractal confirmation (needs 2 subsequent 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price drops below bullish fractal or below 1w EMA50
            if low[i] < bullish_fractal_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above bearish fractal or above 1w EMA50
            if high[i] > bearish_fractal_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for bullish breakout: price > bearish fractal with volume and trend confirmation
            bullish_breakout = (high[i] > bearish_fractal_aligned[i]) and volume_confirmed and (close[i] > ema50_1w_aligned[i])
            # Check for bearish breakout: price < bullish fractal with volume and trend confirmation
            bearish_breakout = (low[i] < bullish_fractal_aligned[i]) and volume_confirmed and (close[i] < ema50_1w_aligned[i])
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals