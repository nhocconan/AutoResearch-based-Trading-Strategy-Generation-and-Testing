#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot reversal with 1-day EMA trend filter and volume confirmation.
# Camarilla pivots provide reversal zones at key levels (L3, H3). 
# Trend filter (1-day EMA34) ensures we trade with the higher timeframe trend.
# Volume confirmation adds conviction to reversal signals.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (long at L3 in uptrend) and bear markets (short at H3 in downtrend).
name = "4h_Camarilla_L3H3_EMA34V_Filter"
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
    
    # Get daily data for pivot and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    # H1 = close + 0.5*(high-low), L1 = close - 0.5*(high-low)
    # Pivot = (high + low + close)/3
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Calculate pivot components
    hl_range = high_d - low_d
    
    # H3 and L3 levels (we focus on these for reversals)
    h3 = close_d + 1.125 * hl_range
    l3 = close_d - 1.125 * hl_range
    
    # Calculate daily EMA34 for trend filter
    close_series = pd.Series(close_d)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily H3, L3, and EMA34 to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price crosses above L3 AND EMA34 uptrend (price > EMA) AND volume confirmation
            long_signal = (close[i] > l3_aligned[i] and close[i-1] <= l3_aligned[i-1]) and \
                          (close[i] > ema34_aligned[i]) and vol_confirm
            
            # Short: price crosses below H3 AND EMA34 downtrend (price < EMA) AND volume confirmation
            short_signal = (close[i] < h3_aligned[i] and close[i-1] >= h3_aligned[i-1]) and \
                           (close[i] < ema34_aligned[i]) and vol_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below L3 OR EMA34 turns down (price < EMA)
            exit_long = (close[i] < l3_aligned[i] and close[i-1] >= l3_aligned[i-1]) or \
                        (close[i] < ema34_aligned[i])
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above H3 OR EMA34 turns up (price > EMA)
            exit_short = (close[i] > h3_aligned[i] and close[i-1] <= h3_aligned[i-1]) or \
                         (close[i] > ema34_aligned[i])
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals