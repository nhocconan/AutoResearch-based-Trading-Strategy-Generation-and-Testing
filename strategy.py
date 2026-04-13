#!/usr/bin/env python3
"""
Hypothesis: 4h 12h Camarilla pivot reversals with 1d volume confirmation and ADX trend filter.
Uses 12h Camarilla pivot levels (H3/L3) for mean-reversion entries, 1d volume spike (>1.5x 20-period average)
to confirm institutional interest, and 12h ADX < 25 to ensure ranging markets where reversals work best.
Long when price crosses above L3 with volume confirmation in ranging market. Short when price crosses below H3.
Exit when price reaches opposite H3/L3 level or mean (P). Target: 20-50 trades/year to avoid fee drag.
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get 12h data for Camarilla pivots and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # Using typical formula: P = (H+L+C)/3, Range = H-L
    # H4 = P + 1.1 * Range * 1.1/2, L4 = P - 1.1 * Range * 1.1/2
    # H3 = P + 1.1 * Range * 1.1/4, L3 = P - 1.1 * Range * 1.1/4
    # H2 = P + 1.1 * Range * 1.1/6, L2 = P - 1.1 * Range * 1.1/6
    # H1 = P + 1.1 * Range * 1.1/12, L1 = P - 1.1 * Range * 1.1/12
    
    # Calculate typical price and range for each period
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Pivot point (P)
    pivot = typical_price_12h
    
    # Camarilla levels
    camarilla_h3 = pivot + 1.1 * range_12h * 1.1 / 4
    camarilla_l3 = pivot - 1.1 * range_12h * 1.1 / 4
    camarilla_p = pivot  # mean/reversion target
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_12h, camarilla_p)
    
    # Calculate ADX for trend strength (ranging market filter)
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    # +DM smoothed, -DM smoothed, TR smoothed over 14 periods
    # DI+ = 100 * smoothed +DM / smoothed TR
    # DI- = 100 * smoothed -DM / smoothed TR
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    # ADX = smoothed DX over 14 periods
    
    # Calculate +DM and -DM
    high_diff = high_12h[1:] - high_12h[:-1]
    low_diff = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Add initial zeros for index alignment
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Calculate TR
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])  # initial seed
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    if len(tr) >= period:
        atr_smoothed = wilders_smooth(tr, period)
        plus_dm_smoothed = wilders_smooth(plus_dm, period)
        minus_dm_smoothed = wilders_smooth(minus_dm, period)
        
        # Avoid division by zero
        di_plus = np.where(atr_smoothed != 0, 100 * plus_dm_smoothed / atr_smoothed, 0)
        di_minus = np.where(atr_smoothed != 0, 100 * minus_dm_smoothed / atr_smoothed, 0)
        
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = wilders_smooth(dx, period)
    else:
        adx = np.zeros_like(tr)
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Ranging market: ADX < 25
    ranging_market = adx_aligned < 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_p_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(ranging_market[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla L3/H3 touch + volume spike + ranging market
        # Long when price crosses above L3 (support bounce)
        long_entry = (close[i-1] <= camarilla_l3_aligned[i-1]) and (close[i] > camarilla_l3_aligned[i])
        # Short when price crosses below H3 (resistance rejection)
        short_entry = (close[i-1] >= camarilla_h3_aligned[i-1]) and (close[i] < camarilla_h3_aligned[i])
        
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        range_confirm = ranging_market[i] > 0.5   # True if ranging market (ADX < 25)
        
        long_signal = long_entry and vol_confirm and range_confirm
        short_signal = short_entry and vol_confirm and range_confirm
        
        # Exit when price reaches pivot (mean) or opposite H3/L3
        exit_long = position == 1 and (close[i] >= camarilla_p_aligned[i] or close[i] >= camarilla_h3_aligned[i])
        exit_short = position == -1 and (close[i] <= camarilla_p_aligned[i] or close[i] <= camarilla_l3_aligned[i])
        
        # Execute signals
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_camarilla_reversal_volume"
timeframe = "4h"
leverage = 1.0