#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d combined with 1w trend filter and volume confirmation
# Fade at R3/S3 levels in ranging markets, breakout continuation at R4/S4 in trending markets
# Uses 1w EMA20 for trend filter to avoid counter-trend trades
# Target: 100-200 total trades over 4 years with controlled risk and low drawdown

name = "6h_camarilla1d_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 1:  # Need previous day's data
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            range_val = ph - pl
            
            camarilla_r4[i] = pc + (range_val * 1.5)
            camarilla_r3[i] = pc + (range_val * 1.25)
            camarilla_s3[i] = pc - (range_val * 1.25)
            camarilla_s4[i] = pc - (range_val * 1.5)
    
    # Align Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume average (24-period for 6h, approx 6 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_ma[i])):
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
            # Exit: price breaks below S3 or trend changes to down
            elif close[i] < s3_6h[i] or close[i] < ema20_1w_aligned[i]:
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
            # Exit: price breaks above R3 or trend changes to up
            elif close[i] > r3_6h[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Determine market regime: trending if price > EMA20_1w, ranging if < EMA20_1w
            is_trending_up = close[i] > ema20_1w_aligned[i]
            is_trending_down = close[i] < ema20_1w_aligned[i]
            
            # Long signals
            if is_trending_up:
                # In uptrend: breakout continuation at R4
                if close[i] > r4_6h[i] and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            else:
                # In downtrend or ranging: fade at S3
                if close[i] < s3_6h[i] and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            
            # Short signals
            if is_trending_down:
                # In downtrend: breakdown continuation at S4
                if close[i] < s4_6h[i] and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                # In uptrend or ranging: fade at R3
                if close[i] > r3_6h[i] and volume[i] > 1.5 * volume_ma[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals