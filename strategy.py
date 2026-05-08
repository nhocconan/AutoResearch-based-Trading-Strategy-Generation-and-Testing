#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + 1w EMA200 trend + 1d Donchian(20) breakout
# In choppy markets (CHOP > 61.8): mean-reversion at Donchian bands
# In trending markets (CHOP < 38.2): trend-following breakouts
# Uses weekly EMA200 for long-term trend filter to avoid counter-trend trades
# Target: 15-25 trades/year to minimize fee drag while capturing major moves

name = "1d_Chop_Regime_Donchian20_EMA200"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate daily Choppiness Index (14-period)
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    
    atr = pd.Series(atr_list).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    maxh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    minl14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = maxh14 - minl14
    
    # Avoid division by zero
    chop = np.full(n, 50.0)
    mask = range14 != 0
    chop[mask] = 100 * np.log10(sum_atr14[mask] / range14[mask]) / np.log10(14)
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_1w_val = ema200_1w_aligned[i]
        chop_val = chop[i]
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        
        if position == 0:
            # Determine regime: choppy (>61.8) or trending (<38.2)
            if chop_val > 61.8:
                # Choppy regime: mean reversion at Donchian bands
                if low_val <= dch_low and close_val > dch_low:
                    # Long signal at lower band bounce
                    signals[i] = 0.25
                    position = 1
                elif high_val >= dch_high and close_val < dch_high:
                    # Short signal at upper band rejection
                    signals[i] = -0.25
                    position = -1
            elif chop_val < 38.2:
                # Trending regime: breakout in direction of weekly trend
                if close_val > dch_high and close_val > ema200_1w_val:
                    # Long breakout above upper band with uptrend
                    signals[i] = 0.25
                    position = 1
                elif close_val < dch_low and close_val < ema200_1w_val:
                    # Short breakdown below lower band with downtrend
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: reverse signal or trend change
            exit_signal = False
            if chop_val > 61.8:
                # In chop: exit at upper band
                if high_val >= dch_high:
                    exit_signal = True
            else:
                # In trend: exit on breakdown or trend reversal
                if close_val < dch_low or close_val < ema200_1w_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: reverse signal or trend change
            exit_signal = False
            if chop_val > 61.8:
                # In chop: exit at lower band
                if low_val <= dch_low:
                    exit_signal = True
            else:
                # In trend: exit on breakout or trend reversal
                if close_val > dch_high or close_val > ema200_1w_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals