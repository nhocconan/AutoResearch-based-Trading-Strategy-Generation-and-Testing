#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Choppiness_Regime_ADX_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Choppiness Index on daily (14-period)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14)
    atr_14 = np.convolve(tr, np.ones(14)/14, mode='same')
    atr_14[:13] = np.nan
    
    # Sum of ATR over 14 periods
    sum_atr_14 = np.nancumsum(atr_14) - np.nancumsum(np.concatenate([np.zeros(13), atr_14[:-13]]))
    sum_atr_14[:13] = np.nan
    
    # Highest high and lowest low over 14 periods
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    for i in range(13, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr_14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)  # avoid division by zero
    
    # ADX(14) on daily
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = np.convolve(tr, np.ones(14)/14, mode='same')
    tr_14[:13] = np.nan
    
    plus_di_14 = 100 * np.convolve(plus_dm, np.ones(14)/14, mode='same') / tr_14
    minus_di_14 = 100 * np.convolve(minus_dm, np.ones(14)/14, mode='same') / tr_14
    plus_di_14[:13] = np.nan
    minus_di_14[:13] = np.nan
    
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    dx = np.where((plus_di_14 + minus_di_14) > 0, dx, 0)
    adx = np.convolve(dx, np.ones(14)/14, mode='same')
    adx[:27] = np.nan  # 13 (for DM) + 13 (for smoothing DX) + 1
    
    # 60-period EMA on daily for trend filter
    ema60_1d = pd.Series(close_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Align daily indicators to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema60_aligned = align_htf_to_ltf(prices, df_1d, ema60_1d)
    
    # 6h EMA20 for entry timing
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema60_aligned[i]) or np.isnan(ema20_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        ema60 = ema60_aligned[i]
        ema20 = ema20_6h[i]
        
        # Regime detection
        is_trending = adx_val > 25
        is_chopping = chop_val > 61.8
        
        if position == 0:
            # Long entry: in trending regime, price above EMA60 and EMA20
            if is_trending and not is_chopping and price > ema60 and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short entry: in trending regime, price below EMA60 and EMA20
            elif is_trending and not is_chopping and price < ema60 and price < ema20:
                signals[i] = -0.25
                position = -1
            # Mean reversion in chopping regime: fade extremes
            elif is_chopping:
                # Long when price touches EMA20 from below in chop
                if price <= ema20 and close[i-1] > ema20:  # crossed below
                    signals[i] = 0.20
                    position = 1
                # Short when price touches EMA20 from above in chop
                elif price >= ema20 and close[i-1] < ema20:  # crossed above
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: trend weakens or reversals
            if adx_val < 20 or price < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakens or reversals
            if adx_val < 20 or price > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines Choppiness Index regime filter with ADX trend strength and EMA crossovers.
# In trending regimes (ADX > 25, CHOP < 61.8): trend following with 60EMA/20EMA alignment.
# In chopping regimes (CHOP > 61.8): mean reversion at 20EMA touchpoints.
# Works in both bull/bear markets by adapting to regime.
# Daily timeframe filters ensure alignment with higher timeframe structure.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.