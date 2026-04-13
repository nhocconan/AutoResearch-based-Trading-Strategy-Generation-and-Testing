#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and ADX trend filter.
# Camarilla levels provide high-probability reversal zones in both trending and ranging markets.
# Volume confirmation ensures reversals have institutional participation.
# ADX filter avoids counter-trend trades in strong trends, favoring mean reversion in ranges.
# Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX (14-period) for trend strength filter on 4h data
    adx_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    # Initial values
    if adx_period < n:
        atr[adx_period] = np.mean(tr[1:adx_period+1])
        plus_di[adx_period] = 100 * np.mean(plus_dm[1:adx_period+1]) / atr[adx_period]
        minus_di[adx_period] = 100 * np.mean(minus_dm[1:adx_period+1]) / atr[adx_period]
        dx[adx_period] = 100 * abs(plus_di[adx_period] - minus_di[adx_period]) / (plus_di[adx_period] + minus_di[adx_period]) if (plus_di[adx_period] + minus_di[adx_period]) > 0 else 0
    
    # Wilder smoothing
    for i in range(adx_period+1, n):
        atr[i] = (atr[i-1] * (adx_period-1) + tr[i]) / adx_period
        plus_di[i] = 100 * ((plus_di[i-1] * (adx_period-1) + plus_dm[i]) / adx_period) / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * ((minus_di[i-1] * (adx_period-1) + minus_dm[i]) / adx_period) / atr[i] if atr[i] > 0 else 0
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) > 0 else 0
    
    adx = np.zeros(n)
    for i in range(2*adx_period, n):
        if i == 2*adx_period:
            adx[i] = np.mean(dx[adx_period:i+1])
        else:
            adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate Camarilla levels from previous day's OHLC
    camarilla_H5 = np.full(n, np.nan)
    camarilla_H4 = np.full(n, np.nan)
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    camarilla_L4 = np.full(n, np.nan)
    camarilla_L5 = np.full(n, np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:
            continue
        # Previous day's OHLC
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        
        # Camarilla calculations
        range_val = prev_high - prev_low
        camarilla_H5[i] = prev_close + 1.1 * range_val * 1.1
        camarilla_H4[i] = prev_close + 1.1 * range_val * 0.55
        camarilla_H3[i] = prev_close + 1.1 * range_val * 0.275
        camarilla_L3[i] = prev_close - 1.1 * range_val * 0.275
        camarilla_L4[i] = prev_close - 1.1 * range_val * 0.55
        camarilla_L5[i] = prev_close - 1.1 * range_val * 1.1
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H5)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_L5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(adx[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_H5_aligned[i]) or np.isnan(camarilla_L5_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx[i]
        
        # Get today's Camarilla levels (aligned to current 4h bar)
        H5 = camarilla_H5_aligned[i]
        H4 = camarilla_H4_aligned[i]
        H3 = camarilla_H3_aligned[i]
        L3 = camarilla_L3_aligned[i]
        L4 = camarilla_L4_aligned[i]
        L5 = camarilla_L5_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        # ADX filter: only trade when ADX < 25 (ranging market) for mean reversion
        ranging_filter = adx_val < 25
        
        if position == 0:
            # Long: price touches L3 or L4 with volume confirmation in ranging market
            if (price <= L3 or price <= L4) and volume_confirm and ranging_filter:
                position = 1
                signals[i] = position_size
            # Short: price touches H3 or H4 with volume confirmation in ranging market
            elif (price >= H3 or price >= H4) and volume_confirm and ranging_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 or H4, or ADX strengthens (trend emerging)
            if (price >= H3 or price >= H4) or adx_val > 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 or L4, or ADX strengthens (trend emerging)
            if (price <= L3 or price <= L4) or adx_val > 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Reversion_Volume_ADX"
timeframe = "4h"
leverage = 1.0