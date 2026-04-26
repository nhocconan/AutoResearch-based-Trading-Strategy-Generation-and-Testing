#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_ATRStop_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h with 1d EMA34 trend filter and ATR-based stoploss. Long when price breaks above R1 with 1d EMA34 uptrend and volume spike; short when price breaks below S1 with 1d EMA34 downtrend and volume spike. Uses discrete position sizing (0.25) to minimize fee drag. Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need 34 for EMA
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12
    pp = (high_1d + low_1d + close_1d) / 3  # Pivot Point (not used in entry but for reference)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for stoploss and volatility filter on 12h
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA (34), ATR (14), volume MA (20)
    start_idx = max(34, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Price breaks above R1 with 1d EMA34 uptrend and volume spike
            long_signal = (close_val > r1_aligned[i]) and (ema_34_aligned[i] > close_1d[-1] if len(close_1d) > 0 else False) and vol_spike
            # Simplify trend check: EMA34 rising (today's EMA > yesterday's EMA)
            # Since we don't have yesterday's EMA in aligned array, use price > EMA as proxy for uptrend
            long_trend = close_val > ema_34_aligned[i]
            
            # Short: Price breaks below S1 with 1d EMA34 downtrend and volume spike
            short_signal = (close_val < s1_aligned[i]) and (ema_34_aligned[i] < close_1d[-1] if len(close_1d) > 0 else False) and vol_spike
            short_trend = close_val < ema_34_aligned[i]
            
            if long_signal and long_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal and short_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit: Price re-enters below R1 OR trailing stop (2.0*ATR below high)
            if (close_val < r1_aligned[i]) or (close_val < highest_since_entry - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit: Price re-enters above S1 OR trailing stop (2.0*ATR above low)
            if (close_val > s1_aligned[i]) or (close_val > lowest_since_entry + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_ATRStop_v2"
timeframe = "12h"
leverage = 1.0