#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversals with 1-day trend filter and volume confirmation
# Camarilla levels from daily timeframe provide precise support/resistance for reversals
# Daily EMA(50) filters trades to align with higher timeframe trend
# Volume > 1.3x average confirms institutional participation at pivot levels
# Works in bull/bear as daily EMA adapts to trend and Camarilla adapts to volatility
# Target: 15-30 trades/year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    rang = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * rang / 2  # Resistance 4
    camarilla_l4 = prev_close - 1.1 * rang / 2  # Support 4
    camarilla_h3 = prev_close + 1.1 * rang / 4  # Resistance 3
    camarilla_l3 = prev_close - 1.1 * rang / 4  # Support 3
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Daily EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to daily EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price touches S3/S4 and reverses up + above daily EMA + volume
            if (close[i] <= l3_aligned[i] * 1.001 and  # Allow small tolerance for touch
                close[i] > l3_aligned[i] and
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price touches R3/R4 and reverses down + below daily EMA + volume
            elif (close[i] >= h3_aligned[i] * 0.999 and  # Allow small tolerance for touch
                  close[i] < h3_aligned[i] and
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 or returns below daily EMA
            if close[i] >= h3_aligned[i] * 0.999 or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 or returns above daily EMA
            if close[i] <= l3_aligned[i] * 1.001 or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Reversal_Volume_v1"
timeframe = "12h"
leverage = 1.0