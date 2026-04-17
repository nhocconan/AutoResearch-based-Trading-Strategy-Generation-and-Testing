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
    
    # Daily data for monthly pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Monthly pivot points (based on previous month)
    # Using previous month's high, low, close
    prev_month_high = np.roll(high_1d, 1)
    prev_month_low = np.roll(low_1d, 1)
    prev_month_close = np.roll(close_1d, 1)
    prev_month_high[0] = high_1d[0]
    prev_month_low[0] = low_1d[0]
    prev_month_close[0] = close_1d[0]
    
    pivot = (prev_month_high + prev_month_low + prev_month_close) / 3
    r1 = 2 * pivot - prev_month_low
    s1 = 2 * pivot - prev_month_high
    r2 = pivot + (prev_month_high - prev_month_low)
    s2 = pivot - (prev_month_high - prev_month_low)
    r3 = prev_month_high + 2 * (pivot - prev_month_low)
    s3 = prev_month_low - 2 * (prev_month_high - pivot)
    
    # Align to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6h RSI for momentum confirmation
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma.iloc[i]
        
        if position == 0:
            # Long: price above R1 with RSI > 50 and volume confirmation
            if price > r1_6h[i] and rsi[i] > 50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price below S1 with RSI < 50 and volume confirmation
            elif price < s1_6h[i] and rsi[i] < 50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price drops below pivot or RSI < 40
            if price < pivot_6h[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price rises above pivot or RSI > 60
            if price > pivot_6h[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MonthlyPivot_R1S1_RSI_Volume"
timeframe = "6h"
leverage = 1.0