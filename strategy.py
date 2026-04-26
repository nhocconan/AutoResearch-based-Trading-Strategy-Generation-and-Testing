#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend
Hypothesis: Use weekly Williams Fractals to identify major swing points on 1d chart.
Long when: price breaks above recent bullish fractal (resistance turned support) in weekly uptrend.
Short when: price breaks below recent bearish fractal (support turned resistance) in weekly downtrend.
Exit on opposite fractal break or weekly trend reversal.
Designed for BTC/ETH: captures major trend continuations with low trade frequency.
Weekly trend filter prevents counter-trend trades in choppy markets.
Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend (needs completed weekly candle)
    weekly_close = df_1w['close'].values
    ema_34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Compute Williams Fractals on weekly data (needs 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values
    )
    # Additional 2-bar delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    fixed_size = 0.25  # 25% position size
    
    # Warmup: need enough data for weekly indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Get aligned weekly values for current daily bar
        weekly_trend = ema_34_1w_aligned[i]
        weekly_close_val = df_1w['close'].values[-1] if len(df_1w['close'].values) > 0 else 0  # placeholder, will be replaced below
        
        # Proper way to get aligned weekly close - we need to align the weekly close series
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
        weekly_close_val = weekly_close_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(weekly_trend) or np.isnan(weekly_close_val) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Flat - look for entry in direction of weekly trend
            # Weekly uptrend: price above EMA34
            # Weekly downtrend: price below EMA34
            if close_val > weekly_trend:  # Weekly uptrend
                # Long: break above recent bullish fractal (resistance turned support)
                if close_val > bullish_fractal_aligned[i]:
                    signals[i] = fixed_size
                    position = 1
            elif close_val < weekly_trend:  # Weekly downtrend
                # Short: break below recent bearish fractal (support turned resistance)
                if close_val < bearish_fractal_aligned[i]:
                    signals[i] = -fixed_size
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on weekly trend reversal or bearish fractal break
            if close_val < weekly_trend or close_val < bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit on weekly trend reversal or bullish fractal break
            if close_val > weekly_trend or close_val > bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0