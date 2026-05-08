#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# Uses daily Camarilla levels (H3, L3) as key support/resistance, with entry when price breaks
# these levels on 12h timeframe, confirmed by daily volume > 1.5x 20-day average.
# In choppy markets (Choppiness Index > 61.8), uses mean-reversion at H3/L3 levels.
# In trending markets (Choppiness Index < 38.2), uses breakout continuation.
# Designed to work in both bull and bear markets by adapting to regime.
# Target: 15-30 trades/year (60-120 total over 4 years).

name = "12h_Camarilla_Pivot_Breakout_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels, volume, and chop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Camarilla levels: H3/L3 = close ± (high-low)*1.1/4
    camarilla_h3 = np.full(len(close_daily), np.nan)
    camarilla_l3 = np.full(len(close_daily), np.nan)
    
    for i in range(1, len(close_daily)):  # start from 1 to use previous day
        high_prev = high_daily[i-1]
        low_prev = low_daily[i-1]
        close_prev = close_daily[i-1]
        camarilla_h3[i] = close_prev + (high_prev - low_prev) * 1.1 / 4
        camarilla_l3[i] = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = np.full(len(close_daily), np.nan)
    vol_sum = 0
    for i in range(len(close_daily)):
        vol_sum += volume[i] if i < len(volume) else 0  # align volumes
        if i >= 19:
            if i == 19:
                vol_avg_20[i] = vol_sum / 20
            else:
                vol_sum -= volume[i-20] if (i-20) < len(volume) else 0
                vol_avg_20[i] = vol_sum / 20
    
    # Calculate daily Choppiness Index (14-period)
    # Chop = 100 * log10(sum(TR) / (max(HH) - min(LL))) / log10(14)
    chop = np.full(len(close_daily), np.nan)
    tr_daily = np.zeros(len(close_daily))
    
    for i in range(len(close_daily)):
        if i == 0:
            tr_daily[i] = high_daily[i] - low_daily[i]
        else:
            tr_daily[i] = max(
                high_daily[i] - low_daily[i],
                abs(high_daily[i] - close_daily[i-1]),
                abs(low_daily[i] - close_daily[i-1])
            )
    
    for i in range(13, len(close_daily)):  # need 14 periods
        # Sum of TR over last 14 periods
        tr_sum = np.sum(tr_daily[i-13:i+1])
        # Highest high and lowest low over last 14 periods
        hh = np.max(high_daily[i-13:i+1])
        ll = np.min(low_daily[i-13:i+1])
        if hh != ll:
            chop[i] = 100 * (np.log10(tr_sum) - np.log10(hh - ll)) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    # Align daily data to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l3)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    
    # Calculate 12h volume average for confirmation
    vol_avg_6 = np.full(n, np.nan)  # 6 periods = 3 days worth
    vol_sum_12h = 0
    for i in range(n):
        vol_sum_12h += volume[i]
        if i >= 5:
            if i == 5:
                vol_avg_6[i] = vol_sum_12h / 6
            else:
                vol_sum_12h -= volume[i-6]
                vol_avg_6[i] = vol_sum_12h / 6
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 6)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_avg_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current daily bar's data (last completed daily bar)
        idx_daily = 0
        while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
            idx_daily += 1
        idx_daily -= 1  # last completed daily bar
        
        if idx_daily < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        camarilla_h3_current = camarilla_h3[idx_daily]
        camarilla_l3_current = camarilla_l3[idx_daily]
        vol_avg_20_current = vol_avg_20[idx_daily]
        chop_current = chop[idx_daily]
        
        if np.isnan(camarilla_h3_current) or np.isnan(camarilla_l3_current) or \
           np.isnan(vol_avg_20_current) or np.isnan(chop_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        vol_current = 0
        if idx_daily < len(volume):
            vol_current = volume[idx_daily]
        vol_confirmed = vol_current > 1.5 * vol_avg_20_current
        
        # Regime detection
        is_choppy = chop_current > 61.8
        is_trending = chop_current < 38.2
        
        # Trading logic
        if position == 0:
            # Look for entry
            if vol_confirmed:
                if is_choppy:
                    # In choppy market: mean reversion at H3/L3 levels
                    if close[i] <= camarilla_l3_current:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] >= camarilla_h3_current:
                        signals[i] = -0.25
                        position = -1
                elif is_trending:
                    # In trending market: breakout continuation
                    if close[i] > camarilla_h3_current:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < camarilla_l3_current:
                        signals[i] = -0.25
                        position = -1
                else:
                    # Transition zone: wait for clearer signal
                    pass
        elif position == 1:
            # Manage long position
            exit_signal = False
            if is_choppy and close[i] >= camarilla_h3_current:
                exit_signal = True  # mean reversion target reached
            elif is_trending and close[i] < camarilla_l3_current:
                exit_signal = True  # breakout failed
            elif not vol_confirmed:
                exit_signal = True  # volume confirmation lost
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Manage short position
            exit_signal = False
            if is_choppy and close[i] <= camarilla_l3_current:
                exit_signal = True  # mean reversion target reached
            elif is_trending and close[i] > camarilla_h3_current:
                exit_signal = True  # breakout failed
            elif not vol_confirmed:
                exit_signal = True  # volume confirmation lost
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals