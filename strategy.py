#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean-reversion with 1d volatility filter
# Uses daily Williams %R(14) to identify oversold/overbought conditions, filtered by daily ATR > 1.5x its 20-period average.
# Enters long when Williams %R < -80 and short when Williams %R > -20, with position sizing of 0.25.
# Designed to capture mean-reversion moves during volatile periods, effective in both bull and bear markets.
# Target: 20-40 trades/year.

name = "4h_WilliamsR_MeanReversion_VolatilityFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR (20-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tr_daily = np.zeros(len(close_daily))
    atr_daily = np.full(len(close_daily), np.nan)
    
    for i in range(len(close_daily)):
        if i == 0:
            tr_daily[i] = high_daily[i] - low_daily[i]
        else:
            tr_daily[i] = max(
                high_daily[i] - low_daily[i],
                abs(high_daily[i] - close_daily[i-1]),
                abs(low_daily[i] - close_daily[i-1])
            )
        
        if i >= 19:
            if i == 19:
                atr_daily[i] = np.mean(tr_daily[:20])
            else:
                atr_daily[i] = (atr_daily[i-1] * 19 + tr_daily[i]) / 20
    
    # Calculate daily Williams %R(14)
    highest_high = np.full(len(high_daily), np.nan)
    lowest_low = np.full(len(low_daily), np.nan)
    williams_r = np.full(len(close_daily), np.nan)
    
    for i in range(len(high_daily)):
        if i >= 13:  # need 14 periods for highest high and lowest low
            highest_high[i] = np.max(high_daily[i-13:i+1])
            lowest_low[i] = np.min(low_daily[i-13:i+1])
            if highest_high[i] != lowest_low[i]:
                williams_r[i] = (highest_high[i] - close_daily[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Align daily data to 4h timeframe
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    
    # Calculate 4h volume average for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current daily bar's ATR (last completed daily bar)
        atr_daily_current = np.nan
        if not np.isnan(atr_daily_aligned[i]):
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                atr_daily_current = atr_daily_aligned[i]
        
        if np.isnan(atr_daily_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate volatility filter: ATR > 1.5x 20-day average ATR
        atr_avg_20 = np.nan
        if idx_daily >= 19:  # need at least 20 days for average
            atr_sum = 0
            count = 0
            for j in range(max(0, idx_daily-19), idx_daily+1):
                if not np.isnan(atr_daily[j]):
                    atr_sum += atr_daily[j]
                    count += 1
            if count >= 10:  # require at least half the period
                atr_avg_20 = atr_sum / count
        
        vol_filter = (not np.isnan(atr_avg_20) and 
                     atr_daily_current > 1.5 * atr_avg_20)
        
        # Check conditions
        williams_value = williams_r_aligned[i]
        
        if position == 0:
            # Look for entry: Williams %R mean-reversion with volatility filter
            if williams_value < -80 and vol_filter:
                signals[i] = 0.25
                position = 1
            elif williams_value > -20 and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral zone or volatility drops
            if (williams_value > -50 or not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral zone or volatility drops
            if (williams_value < -50 or not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals