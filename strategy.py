#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # R3 = H + 1.1*(H-L), S3 = L - 1.1*(H-L)
    # R4 = H + 1.5*(H-L), S4 = L - 1.5*(H-L)
    # Use previous day's H/L for today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_high + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_low - 1.1 * (prev_high - prev_low)
    camarilla_r4 = prev_high + 1.5 * (prev_high - prev_low)
    camarilla_s4 = prev_low - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: current 4h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Volatility filter: avoid extremely low volatility periods
    vol_filter = atr_1d_aligned > (np.nanpercentile(atr_1d_aligned, 30) if not np.all(np.isnan(atr_1d_aligned)) else 0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vol_filt = volume_filter[i]
        vol_filt2 = vol_filter[i]
        
        if position == 0:
            # Enter long: close above R3 + above EMA34 trend + volume filter + volatility filter
            if close[i] > r3 and close[i] > ema34_val and vol_filt and vol_filt2:
                signals[i] = 0.25
                position = 1
            # Enter short: close below S3 + below EMA34 trend + volume filter + volatility filter
            elif close[i] < s3 and close[i] < ema34_val and vol_filt and vol_filt2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 (reversal) or below EMA34 trend
            if close[i] < s3 or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 (reversal) or above EMA34 trend
            if close[i] > r3 or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals