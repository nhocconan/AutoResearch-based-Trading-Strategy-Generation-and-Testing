#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and ADX filter.
# Camarilla levels provide clear support/resistance zones for mean reversion.
# Volume spike confirms momentum at the level touch.
# ADX filter ensures we only trade in ranging markets where reversals work.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR (14-period) for ADX
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr_period = 14
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(atr_period, n):
        plus_di[i] = (100 * np.mean(plus_dm[i-atr_period+1:i+1]) / np.mean(atr[i-atr_period+1:i+1])) if np.mean(atr[i-atr_period+1:i+1]) > 0 else 0
        minus_di[i] = (100 * np.mean(minus_dm[i-atr_period+1:i+1]) / np.mean(atr[i-atr_period+1:i+1])) if np.mean(atr[i-atr_period+1:i+1]) > 0 else 0
    
    dx = np.zeros(n)
    for i in range(atr_period, n):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = (100 * abs(plus_di[i] - minus_di[i]) / di_sum) if di_sum > 0 else 0
    
    adx = np.zeros(n)
    for i in range(2*atr_period, n):
        adx[i] = np.mean(dx[i-atr_period+1:i+1])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate Camarilla levels from previous day's OHLC
    camarilla_H4 = np.full(n, np.nan)
    camarilla_L4 = np.full(n, np.nan)
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    camarilla_H2 = np.full(n, np.nan)
    camarilla_L2 = np.full(n, np.nan)
    camarilla_H1 = np.full(n, np.nan)
    camarilla_L1 = np.full(n, np.nan)
    
    for i in range(1, len(df_1d)):
        day_high = df_1d['high'].iloc[i-1]
        day_low = df_1d['low'].iloc[i-1]
        day_close = df_1d['close'].iloc[i-1]
        range_val = day_high - day_low
        
        camarilla_H4[i] = day_close + range_val * 1.1 / 2
        camarilla_L4[i] = day_close - range_val * 1.1 / 2
        camarilla_H3[i] = day_close + range_val * 1.1 / 4
        camarilla_L3[i] = day_close - range_val * 1.1 / 4
        camarilla_H2[i] = day_close + range_val * 1.1 / 6
        camarilla_L2[i] = day_close - range_val * 1.1 / 6
        camarilla_H1[i] = day_close + range_val * 1.1 / 12
        camarilla_L1[i] = day_close - range_val * 1.1 / 12
    
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H2)
    camarilla_L2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L2)
    camarilla_H1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    camarilla_L1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(adx[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx[i]
        
        camarilla_H4_val = camarilla_H4_aligned[i]
        camarilla_L4_val = camarilla_L4_aligned[i]
        camarilla_H3_val = camarilla_H3_aligned[i]
        camarilla_L3_val = camarilla_L3_aligned[i]
        camarilla_H2_val = camarilla_H2_aligned[i]
        camarilla_L2_val = camarilla_L2_aligned[i]
        camarilla_H1_val = camarilla_H1_aligned[i]
        camarilla_L1_val = camarilla_L1_aligned[i]
        
        # Volume confirmation: current volume > 2x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        # ADX filter: ADX < 25 for ranging market (reversions work better in low trend)
        ranging_filter = adx_val < 25
        
        if position == 0:
            # Long reversal: price touches/slightly breaks L3 or L4 with volume + ranging market
            if ((price <= camarilla_L3_val * 1.002 and price >= camarilla_L4_val * 0.998) or
                (price <= camarilla_L4_val * 1.002 and price >= camarilla_L4_val * 0.998)) and \
               volume_confirm and ranging_filter:
                position = 1
                signals[i] = position_size
            # Short reversal: price touches/slightly breaks H3 or H4 with volume + ranging market
            elif ((price >= camarilla_H3_val * 0.998 and price <= camarilla_H4_val * 1.002) or
                  (price >= camarilla_H4_val * 0.998 and price <= camarilla_H4_val * 1.002)) and \
                 volume_confirm and ranging_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 or H3 level, or volume drops significantly
            if (price >= camarilla_H3_val * 0.998 or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 or L3 level, or volume drops significantly
            if (price <= camarilla_L3_val * 1.002 or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Reversal_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0