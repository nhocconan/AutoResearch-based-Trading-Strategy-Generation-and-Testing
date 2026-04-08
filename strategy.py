#!/usr/bin/env python3
# 4h_camarilla_volume_touch_v1
# Hypothesis: Mean reversion at Camarilla pivot levels with volume confirmation on 4h timeframe.
# In ranging markets: long at S1/S2 with volume surge, short at R1/R2 with volume surge.
# In trending markets: avoid false breakouts using 1d ADX trend filter.
# Uses Camarilla levels from 1d for structure, volume for conviction, ADX for regime.
# Target: 25-50 trades/year (100-200 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_volume_touch_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d Camarilla levels (based on previous day OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = Close + 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low)
    # H2 = Close + 0.75*(High-Low)
    # H1 = Close + 0.5*(High-Low)
    # L1 = Close - 0.5*(High-Low)
    # L2 = Close - 0.75*(High-Low)
    # L3 = Close - 1.125*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    hl_range = prev_high - prev_low
    h4 = prev_close + 1.5 * hl_range
    h3 = prev_close + 1.125 * hl_range
    h2 = prev_close + 0.75 * hl_range
    h1 = prev_close + 0.5 * hl_range
    l1 = prev_close - 0.5 * hl_range
    l2 = prev_close - 0.75 * hl_range
    l3 = prev_close - 1.125 * hl_range
    l4 = prev_close - 1.5 * hl_range
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1d ADX for trend filter (avoid mean reversion in strong trends)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.diff(high)
        minus_dm = np.diff(low)
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = np.abs(np.diff(high))
        tr2 = np.abs(np.diff(low))
        tr3 = np.abs(np.diff(close))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr = np.zeros_like(close)
        atr[period:] = pd.Series(tr).rolling(window=period, min_periods=period).mean().values[period-1:]
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=1).sum().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=1).sum().values / (atr + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(h2_aligned[i]) or np.isnan(h1_aligned[i]) or
            np.isnan(l1_aligned[i]) or np.isnan(l2_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Trend filter: only mean revert when ADX < 25 (ranging market)
        ranging_market = adx_1d_aligned[i] < 25
        
        if position == 1:  # Long position
            # Exit: price reaches H1 or volume dries up
            if close[i] >= h1_aligned[i] or not vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L1 or volume dries up
            if close[i] <= l1_aligned[i] or not vol_surge:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if ranging_market:
                # Long entry: price touches L2/L3/L4 with volume surge
                if vol_surge and (close[i] <= l2_aligned[i] or close[i] <= l3_aligned[i] or close[i] <= l4_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches H2/H3/H4 with volume surge
                elif vol_surge and (close[i] >= h2_aligned[i] or close[i] >= h3_aligned[i] or close[i] >= h4_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals