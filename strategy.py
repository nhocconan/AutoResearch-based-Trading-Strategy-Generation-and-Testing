#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price crossing 6h VWAP with 1d ADX(14) trend filter and volume spike confirmation.
# Uses VWAP as dynamic support/resistance, ADX to ensure trending conditions, and volume spike
# to confirm institutional participation. Designed to capture trend continuations in strong moves
# while avoiding choppy markets. Target: 20-40 trades/year.
name = "6h_VWAP_ADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate VWAP for 6h
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, np.nan)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:10] = np.nan
    vol_ma[-9:] = np.nan
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(adx_aligned[i]) or np.isnan(vwap[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        vwap_val = vwap[i]
        vol_spike_now = vol_spike[i]
        
        if position == 0:
            # Enter long: price above VWAP, ADX > 25 (trending), volume spike
            if price > vwap_val and adx_val > 25 and vol_spike_now:
                signals[i] = 0.25
                position = 1
            # Enter short: price below VWAP, ADX > 25 (trending), volume spike
            elif price < vwap_val and adx_val > 25 and vol_spike_now:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below VWAP OR ADX < 20 (trend weakening)
            if price < vwap_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP OR ADX < 20 (trend weakening)
            if price > vwap_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals