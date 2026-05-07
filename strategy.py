#!/usr/bin/env python3
name = "6h_KeltnerBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA100 trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_100_1d)
    
    # Calculate ATR(20) for Keltner channels
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.inf], tr])  # first value inf
    atr = np.zeros(n)
    for i in range(20, n):
        atr[i] = np.mean(tr[i-20:i])
    
    # Calculate EMA20 for Keltner center line
    ema_20 = np.full(n, np.nan)
    for i in range(20, n):
        ema_20[i] = np.mean(close[i-20:i])
    
    # Keltner channels
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day to prevent overtrading
    
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        trend_up = close > ema_100_1d_aligned[i]
        trend_down = close < ema_100_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Keltner upper with volume in uptrend
            if (close[i] > keltner_upper[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Keltner lower with volume in downtrend
            elif (close[i] < keltner_lower[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Keltner center or trend changes
            if close[i] < ema_20[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Keltner center or trend changes
            if close[i] > ema_20[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, price breaking above/below Keltner channels (EMA20 ± 2*ATR) with volume confirmation and daily trend filter captures institutional order flow. Keltner channels adapt to volatility, providing dynamic support/resistance. The daily EMA100 trend filter ensures trades align with higher timeframe momentum, working in bull markets (breakouts above upper channel in uptrend) and bear markets (breakdowns below lower channel in downtrend) by following the dominant trend. Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while capturing significant moves.