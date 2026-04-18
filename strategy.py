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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels on 1d
    # Pivot = (H+L+C)/3
    # R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    r4_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            # For first bar, use same values (no prior data)
            pivot_1d[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
            range_1d = high_1d[i] - low_1d[i]
        else:
            pivot_1d[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
            range_1d = high_1d[i] - low_1d[i]
        
        if not np.isnan(pivot_1d[i]):
            r1_1d[i] = close_1d[i] + range_1d * 1.1 / 12
            r2_1d[i] = close_1d[i] + range_1d * 1.1 / 6
            r3_1d[i] = close_1d[i] + range_1d * 1.1 / 4
            r4_1d[i] = close_1d[i] + range_1d * 1.1 / 2
            s1_1d[i] = close_1d[i] - range_1d * 1.1 / 12
            s2_1d[i] = close_1d[i] - range_1d * 1.1 / 6
            s3_1d[i] = close_1d[i] - range_1d * 1.1 / 4
            s4_1d[i] = close_1d[i] - range_1d * 1.1 / 2
    
    # Calculate EMA34 on 1d for trend filter
    if len(close_1d) >= 34:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_34_1d = np.full_like(close_1d, np.nan)
    
    # Calculate ATR on 1d for volatility filter
    def calculate_atr(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder smoothing for ATR
        atr = np.full_like(high, np.nan)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align all data to 6h timeframe
    r1_1d_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_1d_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_1d_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 0
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_6h[i]) or np.isnan(r2_1d_6h[i]) or np.isnan(r3_1d_6h[i]) or 
            np.isnan(r4_1d_6h[i]) or np.isnan(s1_1d_6h[i]) or np.isnan(s2_1d_6h[i]) or 
            np.isnan(s3_1d_6h[i]) or np.isnan(s4_1d_6h[i]) or np.isnan(ema_34_1d_6h[i]) or 
            np.isnan(atr_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average on 6h
        # Calculate volume MA on 6h
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_confirm = volume[i] > 1.5 * vol_ma
        else:
            vol_confirm = False
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema_34_1d_6h[i]
        downtrend = close[i] < ema_34_1d_6h[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1d_6h[i] > 0.005 * close[i]  # ATR > 0.5% of price
        
        if position == 0:
            # Long: price breaks above R4 with uptrend and volume
            if close[i] > r4_1d_6h[i] and uptrend and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with downtrend and volume
            elif close[i] < s4_1d_6h[i] and downtrend and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below R3 OR trend reverses
            if close[i] < r3_1d_6h[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above S3 OR trend reverses
            if close[i] > s3_1d_6h[i] or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_R4_S4_Breakout_Volume_EMA34"
timeframe = "6h"
leverage = 1.0