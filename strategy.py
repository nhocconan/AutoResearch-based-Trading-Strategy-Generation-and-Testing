#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price crossing above/below 1d VWAP with volume confirmation (>1.5x 20-bar average) and ADX trend filter (>25).
# VWAP acts as dynamic support/resistance; breaks indicate institutional interest. ADX ensures we only trade in trending markets,
# avoiding whipsaws in chop. Volume surge confirms conviction. Designed for low trade frequency (~20-30/year) to minimize fee decay.
# Works in bull markets (buying VWAP breakouts) and bear markets (selling VWAP breakdowns) by following higher timeframe value.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for VWAP and ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    pv = typical_price * volume_1d
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume_1d)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (using Wilder's smoothing, equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero when both DI are zero
    dx = np.where((di_plus + di_minus) != 0, dx, 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for volume spike detection (on 4h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vwap_val = vwap_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_surge = vol > 1.5 * vol_ma
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_val > 25
        
        if position == 0:
            # Long conditions: price crosses above VWAP + volume surge + trending
            if price > vwap_val and vol_surge and trending:
                signals[i] = 0.25
                position = 1
            # Short conditions: price crosses below VWAP + volume surge + trending
            elif price < vwap_val and vol_surge and trending:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back VWAP OR ADX drops below 20 (trend weakening)
            exit_signal = False
            
            if position == 1:  # long position
                if price < vwap_val or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > vwap_val or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_Breakout_ADX25_VolumeSurge"
timeframe = "4h"
leverage = 1.0