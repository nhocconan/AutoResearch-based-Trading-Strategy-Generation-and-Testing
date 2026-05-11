#!/usr/bin/env python3
# 12h_1dWilliamsFractal_Breakout_Trend
# Hypothesis: Uses daily Williams Fractal breakouts for 12h entries, filtered by 1w trend structure and volume spikes.
# Long when: weekly uptrend (price > 200 EMA), price breaks above daily bearish fractal resistance, and volume > 1.5x 20-period average.
# Short when: weekly downtrend (price < 200 EMA), price breaks below daily bullish fractal support, and volume > 1.5x 20-period average.
# Exit when price reverses back below/above the fractal level or weekly trend breaks.
# Williams Fractals provide key support/resistance levels; breakouts with volume and trend alignment capture strong moves while avoiding false breakouts.
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling breakdowns in downtrends.

name = "12h_1dWilliamsFractal_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Williams Fractals and 1w data for trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 5 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Williams Fractals (requires 2-bar confirmation after center) ---
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Need 2 extra bars for confirmation (Williams Fractal confirmed after 2 following bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # --- 1w trend: price > 200 EMA for uptrend, price < 200 EMA for downtrend ---
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_uptrend = close_1w > ema_200_1w
    weekly_downtrend = close_1w < ema_200_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Williams Fractal (5 bars), EMA200 (200), volume MA (20)
    start_idx = max(5, 200, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1w
        is_weekly_uptrend = weekly_uptrend_aligned[i]
        is_weekly_downtrend = weekly_downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_weekly_uptrend and vol_spike:
                # Long: weekly uptrend + volume spike + price above bearish fractal (resistance breakout)
                if close[i] > bearish_fractal_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_weekly_downtrend and vol_spike:
                # Short: weekly downtrend + volume spike + price below bullish fractal (support breakdown)
                if close[i] < bullish_fractal_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price falls below bullish fractal (support) OR weekly uptrend breaks
                if close[i] < bullish_fractal_aligned[i] or not is_weekly_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above bearish fractal (resistance) OR weekly downtrend breaks
                if close[i] > bearish_fractal_aligned[i] or not is_weekly_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals