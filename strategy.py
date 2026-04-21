#!/usr/bin/env python3
"""
12h_Williams_Fractal_Breakout_1wTrend_v1
Hypothesis: On 12h timeframe, price breaking above recent Williams fractal high or below fractal low indicates institutional breakout, with 1-week EMA50 filter for trend alignment. Uses discrete sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-week EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Williams Fractals on primary timeframe (12h) ===
    high = prices['high'].values
    low = prices['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, prices, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, prices, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_50 = ema_50_1w_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        
        if position == 0:
            # Long: price breaks above bullish fractal + above weekly EMA50
            if price_high > bullish_fractal_val and price_close > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal + below weekly EMA50
            elif price_low < bearish_fractal_val and price_close < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price re-enters fractal level or trend weakens
            if position == 1:
                if price_low < bullish_fractal_val or price_close < ema_50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_high > bearish_fractal_val or price_close > ema_50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1wTrend_v1"
timeframe = "12h"
leverage = 1.0