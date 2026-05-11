#!/usr/bin/env python3
# 4h_WilliamsFractal_EMA34_Retest_Trend
# Hypothesis: Williams fractals on 1d identify key support/resistance zones. Price retesting these levels with EMA34 trend alignment provides high-probability entries. Volume surge confirms institutional interest. Works in bull (buy retests of fractal support in uptrend) and bear (sell retests of fractal resistance in downtrend). Target: 20-40 trades/year.

name = "4h_WilliamsFractal_EMA34_Retest_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get higher timeframe data
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Williams Fractals (need 2-bar confirmation) ---
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Add 2-bar delay for confirmation (fractal forms at bar -2, confirmed at bar 0)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume confirmation (2x 20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 and volume MA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price retests bullish fractal support with volume surge in uptrend
            if (low[i] <= bullish_fractal_aligned[i] * 1.005 and  # allow 0.5% slippage
                close[i] > bullish_fractal_aligned[i] and
                volume_surge and
                ema_34_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price retests bearish fractal resistance with volume surge in downtrend
            elif (high[i] >= bearish_fractal_aligned[i] * 0.995 and  # allow 0.5% slippage
                  close[i] < bearish_fractal_aligned[i] and
                  volume_surge and
                  ema_34_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price breaks below fractal support OR EMA34 turns down
                if (close[i] < bullish_fractal_aligned[i] * 0.995 or  # 0.5% below
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above fractal resistance OR EMA34 turns up
                if (close[i] > bearish_fractal_aligned[i] * 1.005 or  # 0.5% above
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals