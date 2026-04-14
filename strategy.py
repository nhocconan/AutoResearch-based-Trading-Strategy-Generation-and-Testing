#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX Trend + 1d Volatility Breakout
# Use daily True Range expansion (ATR) to detect volatility breakouts
# Trade only when 6h ADX > 25 (trending market) and daily ATR > 1.5x its 20-day average
# Enter long/short based on 6h price breaking above/below prior 6h high/low
# Exit when volatility contracts (ATR < 1.2x average) or trend weakens (ADX < 20)
# Designed for 6h timeframe to capture multi-day trends with volatility filters
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for price action (already the primary timeframe, but we need it for calculations)
    # Actually, prices is already 6h, so we can use it directly
    
    # Load 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR (14 periods) - volatility measure
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr_len = 14
    atr = pd.Series(tr).rolling(window=atr_len, min_periods=atr_len).mean().values
    
    # ATR average (20 periods) for volatility filter
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and ATR_MA to 6h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Calculate 6h ADX for trend strength
    # Using 6h data directly from prices
    adx_len = 14
    
    # True Range for 6h
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[np.nan], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr_6h).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Volatility breakout condition: current ATR > 1.5x ATR average
    vol_expansion = atr_aligned > 1.5 * atr_ma_aligned
    
    # Trend filters
    strong_trend = adx > 25
    weak_trend = adx < 20  # for exit
    
    # Price channels for breakout detection
    # Use 6-period lookback for breakout sensitivity
    lookback = 6
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=1).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=1).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, atr_len, 20, adx_len, lookback)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(atr_ma_aligned[i]) or
            np.isnan(adx[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: price breaks above recent high + volatility expansion + strong trend
            if (close[i] > highest_high[i] and 
                vol_expansion[i] and 
                strong_trend[i]):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below recent low + volatility expansion + strong trend
            elif (close[i] < lowest_low[i] and 
                  vol_expansion[i] and 
                  strong_trend[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: volatility contraction OR trend weakness OR price returns to recent low
            if (not vol_expansion[i] or  # volatility contraction
                weak_trend[i] or         # trend weakening
                close[i] < lowest_low[i]):  # mean reversion
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: volatility contraction OR trend weakness OR price returns to recent high
            if (not vol_expansion[i] or  # volatility contraction
                weak_trend[i] or         # trend weakening
                close[i] > highest_high[i]):  # mean reversion
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ADX_Volatility_Breakout_v1"
timeframe = "6h"
leverage = 1.0