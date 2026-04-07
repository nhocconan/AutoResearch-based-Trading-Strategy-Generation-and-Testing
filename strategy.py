#!/usr/bin/env python3
"""
1h VWAP Reversion + Volume Spike + 4h Trend Filter
- Trend: 4h EMA200 (bull/bear filter)
- Entry: Price deviates >1.5% from 20-period VWAP with volume >1.5x average
- Exit: Price returns to VWAP or trend reversal
- Session filter: 08-20 UTC only
- Target: 15-30 trades/year per symbol (~60-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_vwap_reversion_volume_spike_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC only
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4H TREND FILTER (HTF) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1H VWAP (20-period) ===
    typical_price = (high + low + close) / 3
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not in session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend
        bull_trend = close[i] > ema_4h_aligned[i]
        
        # Calculate VWAP deviation
        vwap_dev = (close[i] - vwap[i]) / vwap[i] * 100
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP OR trend turns bearish
            if vwap_dev >= -0.5 or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to VWAP OR trend turns bullish
            if vwap_dev <= 0.5 or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume spike (>1.5x average)
            if volume[i] <= vol_ma[i] * 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: price deviates significantly from VWAP
            if vwap_dev <= -1.5:  # Price below VWAP -> long
                position = 1
                signals[i] = 0.20
            elif vwap_dev >= 1.5:  # Price above VWAP -> short
                position = -1
                signals[i] = -0.20
    
    return signals