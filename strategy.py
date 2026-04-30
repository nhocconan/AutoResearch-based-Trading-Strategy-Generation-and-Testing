#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR(14) volatility filter.
# Long when price breaks above upper Donchian channel, close > 1d EMA50, and ATR(14) < ATR(50) (low volatility breakout).
# Short when price breaks below lower Donchian channel, close < 1d EMA50, and ATR(14) < ATR(50).
# Exit when price crosses the 10-period EMA of the Donchian midpoint (mean reversion to center).
# Uses Donchian channels for structure, 1d EMA50 for higher timeframe trend, and ATR ratio to avoid high-volatility false breakouts.
# Discrete position sizing at ±0.25 to minimize fee drag while maintaining sufficient exposure.
# Target: 75-150 total trades over 4 years (19-37/year) to stay within proven working range.

name = "4h_Donchian20_1dEMA50_ATR_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # < 1 indicates low volatility regime
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Calculate 10-period EMA of Donchian midpoint for exit signal
    ema_10_mid = pd.Series(donchian_mid).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for ATR50 and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_10_mid[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid_ema = ema_10_mid[i]
        curr_atr_ratio = atr_ratio[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, close > 1d EMA50, low volatility (ATR ratio < 1.0)
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1d and 
                curr_atr_ratio < 1.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, close < 1d EMA50, low volatility (ATR ratio < 1.0)
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1d and 
                  curr_atr_ratio < 1.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below 10-period EMA of Donchian midpoint (mean reversion)
            if curr_close < curr_mid_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above 10-period EMA of Donchian midpoint (mean reversion)
            if curr_close > curr_mid_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals