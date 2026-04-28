#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and ATR-based volatility filter.
# Uses 4h primary timeframe targeting 19-50 trades/year (75-200 total over 4 years).
# 1d EMA34 provides primary trend filter: bull when price > EMA34, bear when price < EMA34.
# Camarilla H3/L3 from 1d provide institutional pivot points with proven edge.
# ATR(20) > 1.5x ATR(50) confirms elevated volatility for breakout validity.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_ATR_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, EMA34 trend, and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0  # H3 = Close + 1.1*(Range)/4
    l3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0  # L3 = Close - 1.1*(Range)/4
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d ATR(20) and ATR(50) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align with original arrays
    atr_20_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA34 and ATR50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr_20_1d_aligned[i]) or
            np.isnan(atr_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h3_1d_aligned[i]
        short_breakout = close[i] < l3_1d_aligned[i]
        
        # Volatility filter: ATR(20) > 1.5x ATR(50) indicates elevated volatility
        vol_filter = atr_20_1d_aligned[i] > 1.5 * atr_50_1d_aligned[i]
        
        long_entry = price_above_ema and long_breakout and vol_filter
        short_entry = price_below_ema and short_breakout and vol_filter
        
        # Exit conditions: opposite Camarilla level (L3/H3 for reversion)
        long_exit = close[i] < l3_1d_aligned[i]  # Exit long at L3
        short_exit = close[i] > h3_1d_aligned[i]  # Exit short at H3
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals