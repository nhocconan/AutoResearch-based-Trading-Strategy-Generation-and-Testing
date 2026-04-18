#!/usr/bin/env python3
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
    
    # Get daily data for Pivot Points and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Classic Pivot Points (based on previous day)
    P = np.full_like(high_1d, np.nan)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    R2 = np.full_like(high_1d, np.nan)
    S2 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        P[i] = (prev_high + prev_low + prev_close) / 3.0
        range_ = prev_high - prev_low
        if range_ > 0:
            R1[i] = P[i] + range_
            S1[i] = P[i] - range_
            R2[i] = P[i] + 2 * range_
            S2[i] = P[i] - 2 * range_
    
    # Calculate 14-period ATR for volatility filter
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all 1d data to 12h timeframe
    P_12h = align_htf_to_ltf(prices, df_1d, P)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 24-period average (moderate)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 14, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(P_12h[i]) or np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or
            np.isnan(R2_12h[i]) or np.isnan(S2_12h[i]) or
            np.isnan(atr_12h[i]) or np.isnan(ema_1w_12h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above weekly EMA for longs, below for shorts
        bullish_bias = close[i] > ema_1w_12h[i]
        bearish_bias = close[i] < ema_1w_12h[i]
        
        if position == 0:
            # Long: price crosses above R1 with volume
            if close[i] > R1_12h[i] and vol_confirm and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 with volume
            elif close[i] < S1_12h[i] and vol_confirm and bearish_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 or ATR expansion (volatility increase)
            if close[i] < S1_12h[i] or atr_12h[i] > 1.5 * atr_12h[i-1]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or ATR expansion
            if close[i] > R1_12h[i] or atr_12h[i] > 1.5 * atr_12h[i-1]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0