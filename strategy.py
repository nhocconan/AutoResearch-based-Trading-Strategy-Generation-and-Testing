#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot breakout with daily volume and volatility filters.
# Camarilla pivot levels (S1, S2, R1, R2) derived from previous day's high/low/close
# provide key support/resistance levels for breakout trading in range-bound and trending markets.
# Volume confirmation ensures breakouts have institutional participation.
# Volatility filter (ATR > 1.5x median ATR) avoids choppy, low-volatility periods.
# Designed for low trade frequency (15-35/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (breakouts above R1/R2) and bear markets (breakouts below S1/S2).
name = "12h_Camarilla_R1S1_Volume_Volatility_Filter"
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
    
    # Get daily data for Camarilla pivots and volatility filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (S1, S2, R1, R2) using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    ph = np.concatenate([[np.nan], high_1d[:-1]])  # Previous high
    pl = np.concatenate([[np.nan], low_1d[:-1]])   # Previous low
    pc = np.concatenate([[np.nan], close_1d[:-1]]) # Previous close
    
    # Calculate pivot point and ranges
    pivot = (ph + pl + pc) / 3.0
    range_hl = ph - pl
    
    # Camarilla levels
    r1 = pc + (range_hl * 1.1 / 12)
    r2 = pc + (range_hl * 1.1 / 6)
    s1 = pc - (range_hl * 1.1 / 12)
    s2 = pc - (range_hl * 1.1 / 6)
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # Calculate median ATR for volatility filter (avoid using mean which is skewed by spikes)
    # Use rolling median of ATR over 30 days
    atr_series = pd.Series(atr)
    atr_median = atr_series.rolling(window=30, min_periods=10).median().values
    
    # Align all daily data to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    atr_median_12h = align_htf_to_ltf(prices, df_1d, atr_median)
    
    # Calculate 24-period average volume for confirmation (2 days of 12h data)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h[i]) or np.isnan(r2_12h[i]) or
            np.isnan(s1_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(atr_median_12h[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_24[i]
        
        # Volatility filter: current ATR must be above 1.5x median ATR
        vol_filter = atr_median_12h[i] > 0 and atr[i] > (1.5 * atr_median_12h[i]) if not np.isnan(atr[i]) else False
        
        if position == 0:
            # Long: price breaks above R1 OR R2 AND volume confirmation AND volatility filter
            long_breakout = close[i] > r1_12h[i] or close[i] > r2_12h[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 OR S2 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and (close[i] < s1_12h[i] or close[i] < s2_12h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR volatility filter fails
            exit_condition = close[i] < s1_12h[i] or not vol_filter
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR volatility filter fails
            exit_condition = close[i] > r1_12h[i] or not vol_filter
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals