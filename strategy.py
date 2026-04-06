#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d ADX trend filter and volume confirmation
# Long when price touches Camarilla L3 support + ADX < 25 (range) + volume > 1.5x average
# Short when price touches Camarilla H3 resistance + ADX < 25 (range) + volume > 1.5x average
# Uses Camarilla levels from previous 1d for mean reversion in ranging markets
# ADX filter avoids trending markets where mean reversion fails
# Target: 50-150 total trades over 4 years with controlled risk

name = "12h_camarilla_reversal_1d_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr = np.zeros_like(high)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    for i in range(period, len(high)):
        if atr[i] != 0:
            plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i])
            minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i])
    
    dx = np.zeros_like(high)
    for i in range(period, len(high)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros_like(high)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation for trend strength
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Camarilla pivot levels from previous 1d
    # Calculate using previous day's OHLC
    camarilla_h3 = np.full_like(close, np.nan)
    camarilla_l3 = np.full_like(close, np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's data
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Camarilla calculations
        range_val = ph - pl
        camarilla_h3_val = pc + (range_val * 1.1 / 4)
        camarilla_l3_val = pc - (range_val * 1.1 / 4)
        
        # These levels are valid for the current 1d bar
        camarilla_h3[i] = camarilla_h3_val
        camarilla_l3[i] = camarilla_l3_val
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price moves above Camarilla H3 or trend strengthens
            elif close[i] > camarilla_h3_aligned[i] or adx_1d_aligned[i] > 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price moves below Camarilla L3 or trend strengthens
            elif close[i] < camarilla_l3_aligned[i] or adx_1d_aligned[i] > 25:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market (ADX < 25)
            # Long: price touches Camarilla L3 support + volume spike
            if (adx_1d_aligned[i] < 25 and
                low[i] <= camarilla_l3_aligned[i] * 1.002 and  # Allow small buffer
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches Camarilla H3 resistance + volume spike
            elif (adx_1d_aligned[i] < 25 and
                  high[i] >= camarilla_h3_aligned[i] * 0.998 and  # Allow small buffer
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals