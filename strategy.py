#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels (based on previous day)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_hl * 1.1 / 4)
    S1 = pivot - (range_hl * 1.1 / 4)
    
    # Align to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Momentum filter: 4h RSI(14) > 50 for long, < 50 for short
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter: current 4h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20, 14)  # Need enough data for EMA34, volume MA, RSI
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        trend = ema34_1d_aligned[i]
        rsi_val = rsi[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above R1 with volume, above trend, and bullish momentum
            if close[i] > r1 and close[i] > trend and rsi_val > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume, below trend, and bearish momentum
            elif close[i] < s1 and close[i] < trend and rsi_val < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (reversion to mean)
            if close[i] < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 (reversion to mean)
            if close[i] > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals