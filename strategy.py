#!/usr/bin/env python3
# 6h_williams_vix_fix_breakout_v1
# Hypothesis: 6h Williams VIX Fix volatility spike + 1d trend filter (EMA200) + volume confirmation.
# Works in bull/bear: VIX Fix identifies volatility expansions (panic/euphoria) that precede reversals;
# 1d EMA200 filters for institutional trend alignment; volume confirms participation. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_vix_fix_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Williams VIX Fix: measures volatility similar to VIX
    # VIX Fix = (Highest Close - Lowest Low) / Highest Close * 100
    # Highest Close = highest close over lookback period
    # Lowest Low = lowest low over lookback period
    lookback = 22  # ~1 month trading days
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    vix_fix = (highest_close - lowest_low) / highest_close * 100
    
    # Align VIX Fix to 6h (already LTF, but ensure proper indexing)
    vix_fix_aligned = vix_fix  # same timeframe
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vix_fix_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: VIX Fix drops below 30 (volatility contraction) OR trend turns bearish
            if vix_fix_aligned[i] < 30.0 or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: VIX Fix drops below 30 OR trend turns bullish
            if vix_fix_aligned[i] < 30.0 or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and high volatility
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            high_volatility = vix_fix_aligned[i] > 50.0  # VIX Fix > 50 indicates extreme volatility
            
            if volume_confirmed and high_volatility:
                # Long: extreme volatility + bullish trend (price above EMA200)
                if close[i] > ema200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: extreme volatility + bearish trend (price below EMA200)
                elif close[i] < ema200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals