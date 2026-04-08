# Hypothetical Strategy: 4h_1d_VWAP_Crossover_with_Volume_Confirmation
# Hypothesis: Price crossing above/below the 1-day VWAP with volume confirmation on 4h timeframe captures institutional
#   flow at key daily value areas. Long when price crosses above 1-day VWAP with above-average volume; short when
#   price crosses below 1-day VWAP with above-average volume. Exit on opposite VWAP cross. This strategy works in
#   both bull and bear markets by following the daily value area, and volume confirmation reduces false signals.
#   Target: 20-50 trades per year (~80-200 total over 4 years) to minimize fee drag.

name = "4h_1d_VWAP_Crossover_with_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate typical price and VWAP for each 1-day bar
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_numerator = (typical_price_1d * df_1d['volume'].values)
    vwap_denominator = df_1d['volume'].values
    
    # Cumulative sums for VWAP (resets each day)
    # We compute VWAP as cumulative typical price * volume / cumulative volume
    # Since we only need the VWAP value at the close of each day, we can compute:
    # VWAP = sum(typical_price * volume) / sum(volume) for the day
    # But for simplicity and to match the 1-day VWAP indicator, we'll use the close-of-day VWAP
    # which is the same as the above for the full day.
    # However, to avoid look-ahead, we need the VWAP value that was known at the close of the previous day.
    # Since VWAP is a cumulative indicator within the day, we cannot know the day's VWAP until the day ends.
    # Therefore, we use the previous day's VWAP value as the reference for the current day.
    # Compute the VWAP for each completed day (using all data up to that day's close)
    cum_sum = np.cumsum(vwap_numerator)
    cum_vol = np.cumsum(vwap_denominator)
    # Avoid division by zero
    vwap_1d = np.where(cum_vol > 0, cum_sum / cum_vol, np.nan)
    # The VWAP value for a day is only known at the close of that day.
    # For signaling on the 4h chart, we want to use the VWAP of the previous completed day.
    # So we shift the VWAP array by 1 to use the prior day's VWAP.
    vwap_1d_prev = np.roll(vwap_1d, 1)
    vwap_1d_prev[0] = np.nan  # First day has no previous day
    
    # Align the previous day's VWAP to 4h timeframe (wait for 1-day bar to close)
    vwap_1d_prev_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_prev)
    
    # Volume confirmation on 4h: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.3
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period: need VWAP and volume MA
    start_idx = max(20, 1) + 1  # at least 20 for volume MA, 1 for VWAP (but shifted)
    
    for i in range(start_idx, n):
        # Skip if VWAP is not available
        if np.isnan(vwap_1d_prev_aligned[i]):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals during session with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 1-day VWAP (previous day's VWAP)
            if close[i] < vwap_1d_prev_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1-day VWAP (previous day's VWAP)
            if close[i] > vwap_1d_prev_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price crosses above 1-day VWAP with volume confirmation
            if close[i] > vwap_1d_prev_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below 1-day VWAP with volume confirmation
            elif close[i] < vwap_1d_prev_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals