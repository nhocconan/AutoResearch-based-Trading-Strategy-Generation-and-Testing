#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly high/low/close from daily data (resample weekly using last Friday)
    # Since we cannot resample, we approximate weekly pivot using rolling window of 5 days
    # Weekly high = max of last 5 daily highs
    # Weekly low = min of last 5 daily lows
    # Weekly close = close of 5th day ago (Friday)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low using 5-day window (approximation of weekly)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values  # last of 5
    
    # Calculate weekly pivot points (classic formula)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 12h timeframe (using Friday's close as anchor)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Load 12h data for trend filter (using 12h itself as HTF is not needed for trend)
    # We'll use 12h EMA50 as trend filter
    close_12h = prices['close'].values
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR for volatility filter (using 12h data)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Price array
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema50_val = ema50[i]
        atr_val = atr_14[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volatility filter: ATR > 20-period average ATR (avoid low volatility chop)
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = atr_val > 0.5 * atr_ma_20  # Reduced threshold to allow more trades
        
        # Volume filter: current volume > 1.2 * 20-period average volume (more sensitive)
        vol_spike = vol > 1.2 * vol_ma
        
        # Trend filter: price above/below 12h EMA50
        uptrend = price > ema50_val
        downtrend = price < ema50_val
        
        if position == 0:
            # Long: price crosses above S1 (support) + uptrend + volatility filter + volume spike
            if price > s1_val and uptrend and vol_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 (resistance) + downtrend + volatility filter + volume spike
            elif price < r1_val and downtrend and vol_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through pivot or volatility drops or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below pivot or volatility collapse or volume drop
                if price < pivot_val or not vol_filter or not vol_spike:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above pivot or volatility collapse or volume drop
                if price > pivot_val or not vol_filter or not vol_spike:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WeeklyPivot_S1_R1_EMA50_ATRVolFilter_VolSpike"
timeframe = "12h"
leverage = 1.0