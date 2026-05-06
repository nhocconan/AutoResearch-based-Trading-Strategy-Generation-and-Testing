#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation and 1-day EMA trend filter
# Long when price crosses above Camarilla R1 with volume > 1.3x average and price > 1-day EMA34
# Short when price crosses below Camarilla S1 with volume > 1.3x average and price < 1-day EMA34
# Uses daily EMA for trend filter and Camarilla levels for precise entry/exit
# Designed to work in bull markets via R1 breakouts and in bear markets via S1 breakdowns
# Target: 20-40 trades per year (80-160 over 4 years) with 0.30 position sizing

name = "4h_Camarilla_R1S1_1dEMA34_Volume_v1"
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
    
    # Calculate 12h Camarilla pivot levels (R1, S1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Typical price for pivot calculation
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    pivot = typical_price.rolling(window=1, min_periods=1).mean().values  # Current bar's typical price
    high_val = df_12h['high'].values
    low_val = df_12h['low'].values
    
    # Camarilla R1 and S1
    r1 = pivot + (high_val - low_val) * 1.1 / 12
    s1 = pivot - (high_val - low_val) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # 1-day EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after first bar for EMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price crosses above R1 with volume confirmation and uptrend
            if close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and volume_filter[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short entry: price crosses below S1 with volume confirmation and downtrend
            elif close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and volume_filter[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 (strong reversal)
            if close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above R1 (strong reversal)
            if close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals