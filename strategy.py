#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly volatility regime filter
# Uses Donchian(20) breakout for trend direction and weekly ATR-based volatility regime
# to avoid trading in high volatility chop. Designed for low trade frequency (<20/year)
# to minimize fee drag. Works in bull markets via trend following and in bear markets
# by avoiding false breakouts during volatile ranging periods.

name = "daily_donchian20_weekly_vol_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR(10) for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_1w_ma = pd.Series(atr_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1w_ratio = np.where(atr_1w_ma > 0, atr_1w / atr_1w_ma, 1.0)
    
    # Align weekly volatility ratio to daily
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_ratio)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when volatility is low (ratio < 1.2)
        low_vol_regime = atr_ratio_aligned[i] < 1.2
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Using previous bar's high
        breakout_down = close[i] < lowest_low[i-1]   # Using previous bar's low
        
        # Long: bullish breakout in low volatility regime
        if breakout_up and low_vol_regime:
            signals[i] = 0.25
        # Short: bearish breakout in low volatility regime
        elif breakout_down and low_vol_regime:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals