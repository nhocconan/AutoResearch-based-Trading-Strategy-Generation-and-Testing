#!/usr/bin/env python3
"""
6h_MultiTimeframe_Donchian_Breakout_12hTrend_Filter
Hypothesis: 6-hour Donchian(20) breakouts in the direction of the 12-hour trend (EMA34) yield high-probability moves. Volume confirmation and ATR volatility filter reduce false signals. Works in bull/bear by only taking breakouts aligned with the 12h EMA trend, avoiding counter-trend trades during reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h Donchian(20) channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate rolling max/min for Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume and volatility filters
    volume = prices['volume'].values
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate ATR(14) for volatility filter
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required values are not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(ema_12h_aligned[i])):
            continue
        
        # Volume filter: current volume > 20-period average
        volume_filter = volume[i] > vol_ma[i]
        
        # Volatility filter: ATR > 0 (avoid degenerate cases)
        vol_filter = atr[i] > 0
        
        # Trend filter: price above/below 12h EMA34
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Entry logic
        if volume_filter and vol_filter:
            # Long: upward breakout + price above 12h EMA (uptrend)
            if breakout_up and price_above_ema:
                signals[i] = 0.25
            # Short: downward breakout + price below 12h EMA (downtrend)
            elif breakout_down and price_below_ema:
                signals[i] = -0.25
    
    return signals

name = "6h_MultiTimeframe_Donchian_Breakout_12hTrend_Filter"
timeframe = "6h"
leverage = 1.0