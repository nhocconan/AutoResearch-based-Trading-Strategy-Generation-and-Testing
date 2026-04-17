#!/usr/bin/env python3
"""
Hypothesis: 1h VWAP Deviation with 4h Trend Filter and Session Filter.
Long when price > VWAP(20) and 4h close > 4h EMA50 during 08-20 UTC.
Short when price < VWAP(20) and 4h close < 4h EMA50 during 08-20 UTC.
Exit when price crosses VWAP or 4h trend reverses.
Uses 4h for trend direction, 1h for VWAP-based entry timing and session filter.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h VWAP (20-period)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    # Rolling VWAP: reset every 20 periods using windowed approach
    vwap_20 = np.full(n, np.nan)
    for i in range(19, n):
        start_idx = i - 19
        window_pv = np.sum(pv[start_idx:i+1])
        window_vol = np.sum(volume[start_idx:i+1])
        if window_vol > 0:
            vwap_20[i] = window_pv / window_vol
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for 4h EMA50 and 1h VWAP20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vwap_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap_20[i]
        ema50_4h_val = ema50_4h_aligned[i]
        
        if position == 0:
            # Long: price > VWAP and 4h close > 4h EMA50
            if price > vwap_val and close_4h[-1] > ema50_4h_val if len(close_4h) > 0 else False:
                # Need to get current 4h close - use the last value from aligned 4h close
                df_4h_close = get_htf_data(prices, '4h')['close'].values
                df_4h_close_aligned = align_htf_to_ltf(prices, df_4h, df_4h_close)
                if price > vwap_val and df_4h_close_aligned[i] > ema50_4h_val:
                    signals[i] = 0.20
                    position = 1
            # Short: price < VWAP and 4h close < 4h EMA50
            elif price < vwap_val and df_4h_close_aligned[i] < ema50_4h_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price < VWAP OR 4h close < 4h EMA50
            if price < vwap_val or df_4h_close_aligned[i] < ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price > VWAP OR 4h close > 4h EMA50
            if price > vwap_val or df_4h_close_aligned[i] > ema50_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VWAP_4hEMA50_Session"
timeframe = "1h"
leverage = 1.0