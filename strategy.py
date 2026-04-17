#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Pivot Point breakout with weekly EMA trend filter and volume confirmation.
# Uses daily pivot levels (PP, R1, S1) for entries, weekly EMA for trend filter, volume for confirmation.
# Designed to work in bull (breakouts with trend) and bear (mean reversion via pivot rejects).
# Target: 10-25 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily pivot points (PP, R1, S1)
    # Standard pivot: PP = (high + low + close) / 3
    # R1 = 2*PP - low, S1 = 2*PP - high
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    
    # Align weekly EMA and daily pivots to 1d
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average (reduces trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need weekly EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to balance trades)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > r1_1d_aligned[i]
        price_below_s1 = close[i] < s1_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume and above weekly EMA34
            if (price_above_r1 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume and below weekly EMA34
            elif (price_below_s1 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily PP OR below weekly EMA34
            if (close[i] < pp_1d_aligned[i]) or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily PP OR above weekly EMA34
            if (close[i] > pp_1d_aligned[i]) or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DailyPivot_Breakout_WeeklyEMA34_Volume"
timeframe = "1d"
leverage = 1.0