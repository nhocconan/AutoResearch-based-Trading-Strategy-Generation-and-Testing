#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and weekly EMA50 trend filter
# Uses 6h primary timeframe to target 50-150 total trades over 4 years (12-37/year)
# Donchian breakout captures momentum moves; ATR filter ensures sufficient volatility
# Weekly EMA50 trend filter aligns with higher timeframe momentum to avoid counter-trend trades
# Designed for both bull and bear markets: trend filter works in bull, ATR filter adapts to volatility regimes
# Tight entry conditions to minimize fee drag while maintaining edge

name = "6h_Donchian20_1dATR_Filter_1wEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar TR
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar TR vs previous close (use open as prev close)
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First bar TR vs previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 1d ATR moving average (20-period) for relative volatility filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, ATR, and HTF alignment)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: current ATR > 1.2 * ATR MA (ensures sufficient volatility)
        volatilty_filter = atr_14_aligned[i] > (atr_ma_20_aligned[i] * 1.2)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper + volatility filter + price > weekly EMA50
            if close[i] > highest_high[i] and volatilty_filter and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + volatility filter + price < weekly EMA50
            elif close[i] < lowest_low[i] and volatilty_filter and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower (reversal signal)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper (reversal signal)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals