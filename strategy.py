#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with daily volume filter and trend filter.
# Camarilla levels (H4/L4) act as strong support/resistance levels.
# Price rejection at these levels with volume confirmation provides high-probability reversals.
# Daily volume filter ensures we only trade when institutional participation is present.
# Trend filter (daily EMA34) ensures we trade in direction of higher timeframe trend.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (long at L4 in uptrend) and bear markets (short at H4 in downtrend).
name = "4h_Camarilla_H4L4_Volume_EMA34"
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
    
    # Get daily data for Camarilla levels and filters (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (H4, L4) using previous day's data
    # H4 = Close + 1.1/2 * (High - Low)
    # L4 = Close - 1.1/2 * (High - Low)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    camarilla_H4 = close_d + 1.1/2 * (high_d - low_d)
    camarilla_L4 = close_d - 1.1/2 * (high_d - low_d)
    
    # Calculate daily EMA34 for trend filter
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False).values
    
    # Align Camarilla levels and EMA34 to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
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
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema34_aligned[i]
        price_below_ema = close[i] < ema34_aligned[i]
        
        if position == 0:
            # Long: price at L4 with rejection (close > L4) AND volume confirmation AND uptrend
            long_setup = close[i] > camarilla_L4_aligned[i]
            if vol_confirm and price_above_ema and long_setup:
                signals[i] = 0.25
                position = 1
            # Short: price at H4 with rejection (close < H4) AND volume confirmation AND downtrend
            elif vol_confirm and price_below_ema and close[i] < camarilla_H4_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below L4 OR trend changes to downtrend
            exit_condition = close[i] < camarilla_L4_aligned[i] or not price_above_ema
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above H4 OR trend changes to uptrend
            exit_condition = close[i] > camarilla_H4_aligned[i] or not price_below_ema
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals