#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
# Uses 20-period Donchian channels on 4h, filtered by daily ATR-based volatility regime and volume > 1.5x 20-period average.
# Designed to capture breakouts in high-volatility trending markets while avoiding low-volatility false breakouts.
# Works in bull/bear via ATR filter targeting periods of increased volatility. Target: 25-50 trades/year (100-200 total over 4 years).

name = "4h_Donchian20_1dATRFilter_VolumeConfirm"
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
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            highest_high_20[i] = np.max(high[i-20:i])
            lowest_low_20[i] = np.min(low[i-20:i])
    
    # Get daily data for ATR calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
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
        
        if i >= 13:
            if i == 13:
                atr_daily[i] = np.mean(tr_daily[:14])
            else:
                atr_daily[i] = (atr_daily[i-1] * 13 + tr_daily[i]) / 14
    
    # Calculate 4h volume average for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily ATR to 4h timeframe
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(atr_daily_aligned[i]) or np.isnan(vol_avg_20[i])):
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
        price_above_upper = close[i] > highest_high_20[i]
        price_below_lower = close[i] < lowest_low_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with volatility filter and volume confirmation
            if price_above_upper and vol_filter and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            elif price_below_lower and vol_filter and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower band or volatility/volume drops
            if (close[i] < lowest_low_20[i] or not vol_filter or 
                volume[i] <= 1.5 * vol_avg_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper band or volatility/volume drops
            if (close[i] > highest_high_20[i] or not vol_filter or 
                volume[i] <= 1.5 * vol_avg_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals