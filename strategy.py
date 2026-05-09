#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot point reversal with 1d trend filter and volume confirmation.
# Uses daily Camarilla levels (H4/L4) for mean reversion entries in ranging markets,
# filtered by 1-day ADX < 30 to avoid trending markets, and volume spike for confirmation.
# Designed to work in both bull (buy at L4 in range) and bear (sell at H4 in range).
# Target: 15-30 trades/year to minimize fee drag.
name = "12h_Camarilla_H4L4_1dADX30_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (H4, L4)
    # Formula: H4 = Close + 1.5 * (High - Low), L4 = Close - 1.5 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Calculate 14-period ADX for daily timeframe (trend filter)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    plus_di = 100 * np.where(atr > 0, 
                             np.convolve(plus_dm, np.ones(period)/period, mode='full')[:len(plus_dm)] / atr, 0)
    minus_di = 100 * np.where(atr > 0,
                              np.convolve(minus_dm, np.ones(period)/period, mode='full')[:len(minus_dm)] / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < period:
            adx[i] = np.nan
        elif i == period:
            adx[i] = np.mean(dx[1:period+1])
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align 1d Camarilla levels and ADX to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data for Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price at or below L4 + 1d ADX < 30 (ranging) + volume spike
            if (price <= camarilla_l4_aligned[i] and adx_aligned[i] < 30 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price at or above H4 + 1d ADX < 30 (ranging) + volume spike
            elif (price >= camarilla_h4_aligned[i] and adx_aligned[i] < 30 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches midpoint (Camarilla H3/L3) or ADX rises above 40 (trend)
            camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)  # H3 level
            camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)  # L3 level
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if price >= midpoint or adx_aligned[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches midpoint or ADX rises above 40 (trend)
            camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)  # H3 level
            camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)  # L3 level
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if price <= midpoint or adx_aligned[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals