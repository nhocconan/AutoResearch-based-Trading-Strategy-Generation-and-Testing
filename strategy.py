#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot (R1/S1) breakout with daily volume confirmation and 1h EMA34 trend filter.
# Camarilla pivots provide mathematically derived support/resistance levels based on prior day's range.
# Breakouts above R1 or below S1 with volume confirmation indicate institutional participation.
# 1h EMA34 filter ensures we only trade in the direction of short-term trend to avoid counter-trend whipsaws.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakouts below S1 in downtrend).
name = "4h_Camarilla_R1S1_Breakout_Volume_EMA34Filter"
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
    
    # Get daily data for Camarilla pivots (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for today using yesterday's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Calculate pivot range
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r1 = close_prev + range_prev * 1.1 / 12
    s1 = close_prev - range_prev * 1.1 / 12
    
    # Align to 4h timeframe (values available after daily candle closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1h data for EMA34 trend filter
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate EMA34 on 1h closes
    close_1h = df_1h['close'].values
    ema_34_1h = pd.Series(close_1h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1h EMA34 to 4h timeframe
    ema_34_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_34_1h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Trend filter: price above/below EMA34
        price_above_ema = close[i] > ema_34_1h_aligned[i]
        price_below_ema = close[i] < ema_34_1h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND price above EMA34
            long_breakout = close[i] > r1_aligned[i]
            if vol_confirm and price_above_ema and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND price below EMA34
            elif vol_confirm and price_below_ema and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR price drops below EMA34 (trend change)
            exit_condition = close[i] < s1_aligned[i] or close[i] < ema_34_1h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR price rises above EMA34 (trend change)
            exit_condition = close[i] > r1_aligned[i] or close[i] > ema_34_1h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals