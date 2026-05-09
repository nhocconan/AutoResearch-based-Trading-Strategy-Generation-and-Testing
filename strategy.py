#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pullback_1dTrend_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily ATR-based volatility filter (low volatility preferred)
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    prev_close = np.roll(df_1d['close'], 1)
    prev_high = np.roll(df_1d['high'], 1)
    prev_low = np.roll(df_1d['low'], 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla levels: H3, L3 (tighter levels for pullbacks)
    H3 = (prev_high + prev_low) * 1.1 / 2 - (prev_high - prev_low) * 1.1 / 4
    L3 = (prev_high + prev_low) * 1.1 / 2 + (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ATR MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        atr_14_val = atr_14_aligned[i]
        atr_ma_50_val = atr_ma_50_aligned[i]
        
        # Volatility filter: only trade when volatility is below average (avoid choppy markets)
        vol_filter = atr_14_val < atr_ma_50_val
        
        if position == 0:
            # Enter long: Pullback to L3 in uptrend (price > EMA34) + low volatility
            if vol_filter and close[i] <= l3 and close[i] > ema_1d:
                signals[i] = 0.25
                position = 1
            # Enter short: Pullback to H3 in downtrend (price < EMA34) + low volatility
            elif vol_filter and close[i] >= h3 and close[i] < ema_1d:
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