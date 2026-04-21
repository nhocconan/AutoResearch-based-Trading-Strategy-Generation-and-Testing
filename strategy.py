#!/usr/bin/env python3
"""
12h_1d_VWAP_Deviation_MeanReversion
Hypothesis: Price deviates significantly from daily VWAP and reverts back with volume confirmation. 
Works in both bull and bear markets by fading extremes at VWAP ± 2*ATR(20) with volume filter.
Target: 12-37 trades/year to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for VWAP and ATR
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate daily VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (high_daily + low_daily + close_daily) / 3.0
    pv = typical_price * volume_daily
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume_daily)
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, typical_price)
    
    # Calculate daily ATR(20)
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 20:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-20:i])
    
    # Align daily VWAP and ATR to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_daily, vwap)
    atr_aligned = align_htf_to_ltf(prices, df_daily, atr)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_val = vwap_aligned[i]
        atr_val = atr_aligned[i]
        vol_ok = volume_filter[i]
        
        upper_band = vwap_val + (2.0 * atr_val)
        lower_band = vwap_val - (2.0 * atr_val)
        
        if position == 0:
            # Long: price touches/bounces off lower band with volume
            if price <= lower_band and vol_ok:
                # Confirmation: closing in upper half of bar
                if close[i] > (high[i] + low[i]) / 2:
                    signals[i] = 0.25
                    position = 1
            # Short: price touches/rejects upper band with volume
            elif price >= upper_band and vol_ok:
                # Confirmation: closing in lower half of bar
                if close[i] < (high[i] + low[i]) / 2:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP or hits upper band
            if price >= vwap_val or price >= upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to VWAP or hits lower band
            if price <= vwap_val or price <= lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_VWAP_Deviation_MeanReversion"
timeframe = "12h"
leverage = 1.0