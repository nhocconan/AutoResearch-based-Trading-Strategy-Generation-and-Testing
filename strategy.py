#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pullback_1dTrend_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR(14) for volume filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily volume MA(20) for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    prev_close = np.roll(df_1d['close'], 1)
    prev_high = np.roll(df_1d['high'], 1)
    prev_low = np.roll(df_1d['low'], 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla levels: H3, L3 (more reliable than H4/L4 for pullbacks)
    H3 = (prev_high + prev_low) * 1.1 / 2 - (prev_high - prev_low) * 1.1 / 4
    L3 = (prev_high + prev_low) * 1.1 / 2 + (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need 34 for EMA + 1 for roll
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        atr_14 = atr_14_1d_aligned[i]
        vol_ma_20 = vol_ma_20_1d_aligned[i]
        vol_current = volume[i * 2] if i * 2 < len(volume) else volume[-1]  # Approximate 12h vol from 5m data
        
        # Volume filter: current volume > 1.5x 20-day average (only for 12h bar)
        vol_filter = vol_current > 1.5 * vol_ma_20
        
        if position == 0:
            # Enter long: Pullback to L3 in uptrend (price > EMA34) with volume confirmation
            if close[i] <= l3 and close[i] > ema_1d and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Pullback to H3 in downtrend (price < EMA34) with volume confirmation
            elif close[i] >= h3 and close[i] < ema_1d and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below EMA34 (trend change) or reaches H3 (target)
            if close[i] < ema_1d or close[i] >= h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above EMA34 (trend change) or reaches L3 (target)
            if close[i] > ema_1d or close[i] <= l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals