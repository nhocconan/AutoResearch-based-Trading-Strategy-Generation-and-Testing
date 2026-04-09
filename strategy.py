#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter
# Donchian breakouts capture momentum in trending markets (bull/bear)
# Volume confirmation ensures breakout validity
# Chop regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) adapts to market conditions
# In trending regime (CHOP < 38.2): trade Donchian breakouts
# In ranging regime (CHOP > 61.8): fade Donchian touches (mean reversion)
# Position size 0.25 to limit drawdown
# Target: 100-200 total trades over 4 years (25-50/year) for optimal fee balance

name = "4h_1d_donchian_chop_vol_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # ATR(14) - sum of TR
    atr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        else:
            atr_1d[i] = np.nansum(tr_1d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_1d = np.zeros(len(df_1d))
    ll_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 14:
            hh_1d[i] = np.nan
            ll_1d[i] = np.nan
        else:
            hh_1d[i] = np.nanmax(high_1d[i-13:i+1])
            ll_1d[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Chopiness Index: CHOP = 100 * log10(ATR(14) / (HH(14) - LL(14))) / log10(14)
    chop_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 14 or np.isnan(hh_1d[i]) or np.isnan(ll_1d[i]) or hh_1d[i] == ll_1d[i]:
            chop_1d[i] = np.nan
        else:
            chop_1d[i] = 100 * np.log10(atr_1d[i] / (hh_1d[i] - ll_1d[i])) / np.log10(14)
    
    # Align 1d chop to 4h timeframe
    chop_1d_4h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Donchian channels on 4h (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.nanmax(high[i-19:i+1])
            lowest_low[i] = np.nanmin(low[i-19:i+1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.nanmean(volume[i-19:i+1])
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(chop_1d_4h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_1d_4h[i]
        vol_conf = volume_confirm[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if chop < 38.2:  # Trending regime
                # Exit when price touches or crosses below Donchian lower
                if close[i] <= lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif chop > 61.8:  # Ranging regime
                # Exit when price returns to midpoint (mean reversion)
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] >= midpoint:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Transition regime
                # Hold position
                signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop < 38.2:  # Trending regime
                # Exit when price touches or crosses above Donchian upper
                if close[i] >= highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif chop > 61.8:  # Ranging regime
                # Exit when price returns to midpoint (mean reversion)
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] <= midpoint:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Transition regime
                # Hold position
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if not vol_conf:
                signals[i] = 0.0
                continue
                
            if chop < 38.2:  # Trending regime - trade breakouts
                # Go long when price breaks above Donchian upper
                # Go short when price breaks below Donchian lower
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
            elif chop > 61.8:  # Ranging regime - fade extremes (mean reversion)
                # Go long when price touches Donchian lower and shows rejection
                # Go short when price touches Donchian upper and shows rejection
                if i > 30:
                    # Long: price near lower band and closing higher (bullish rejection)
                    if close[i] <= lowest_low[i] * 1.001 and close[i] > close[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price near upper band and closing lower (bearish rejection)
                    elif close[i] >= highest_high[i] * 0.999 and close[i] < close[i-1]:
                        position = -1
                        signals[i] = -0.25
            else:  # Transition regime - no clear signal
                signals[i] = 0.0
    
    return signals