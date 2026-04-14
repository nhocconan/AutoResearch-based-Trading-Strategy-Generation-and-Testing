#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d ADX trend filter and volume confirmation
# Camarilla pivot levels (S3/S4 for long, R3/R4 for short) identify reversal zones in ranging markets
# 1d ADX(14) < 30 filters for low-volatility ranging markets where reversals are more reliable
# Volume > 1.3x 20-period EMA confirms participation at pivot levels
# Target: 20-30 trades/year with mean-reversion logic suited for 2025 bear/range conditions
# Stops via opposite pivot touch to avoid whipsaws

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    # Width = H - L
    # S3 = C - (H - L) * 1.1/2
    # S4 = C - (H - L) * 1.1
    # R3 = C + (H - L) * 1.1/2
    # R4 = C + (H - L) * 1.1
    
    # Shift to use previous day's data
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    width = prev_high - prev_low
    
    s3 = typical_price - width * 1.1 / 2.0
    s4 = typical_price - width * 1.1
    r3 = typical_price + width * 1.1 / 2.0
    r4 = typical_price + width * 1.1
    
    # Calculate 1d ADX (14-period) for trend filter (low ADX = ranging)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(1, n):
        # Get aligned 1d ADX
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        if np.isnan(s3[i]) or np.isnan(s4[i]) or np.isnan(r3[i]) or np.isnan(r4[i]) or \
           np.isnan(adx_1d_aligned) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        # Trend filter: ADX < 30 indicates ranging market (better for mean reversion)
        ranging = adx_1d_aligned < 30
        
        if position == 0:  # No position - look for reversal entries at pivot levels
            # Long reversal: price touches/surpasses S3/S4 with volume in ranging market
            if low[i] <= s3[i] and volume_confirm and ranging:
                position = 1
                signals[i] = position_size
            # Short reversal: price touches/surpasses R3/R4 with volume in ranging market
            elif high[i] >= r4[i] and volume_confirm and ranging:
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit at opposite pivot or reversal
            # Exit if price reaches R3 (opposite pivot level)
            if high[i] >= r3[i]:
                position = 0
                signals[i] = 0.0
            # Optional: exit if price closes back above S3 (failed reversal)
            elif close[i] > s3[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit at opposite pivot or reversal
            # Exit if price reaches S3 (opposite pivot level)
            if low[i] <= s3[i]:
                position = 0
                signals[i] = 0.0
            # Optional: exit if price closes back below R4 (failed reversal)
            elif close[i] < r4[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_1dADX_RangeRev"
timeframe = "4h"
leverage = 1.0