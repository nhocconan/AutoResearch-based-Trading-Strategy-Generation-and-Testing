#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal + 1d trend filter
# Camarilla pivot levels: R4 = C + 1.1*(H-L), R3 = C + 1.1*(H-L)/2, etc.
# In trending markets (price > 1d EMA50): breakout above R4 or below S4 continues trend.
# In ranging markets (price near 1d EMA50): reversals at R3/S3.
# Uses volume confirmation to avoid false breakouts.
# Target: 80-180 total trades over 4 years.

name = "6h_camarilla_1d_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA50 calculation for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot levels based on previous day's OHLC
    camarilla_r4 = close_1d + 1.1 * (high_1d - low_1d) * 1.1/2  # Actually: C + 1.1*(H-L)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_s4 = close_1d - 1.1 * (high_1d - low_1d) * 1.1/2  # Actually: C - 1.1*(H-L)
    
    # Correct calculation
    camarilla_r4 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_s4 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if required data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S3 in uptrend or reaches R4
            elif close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R3 in downtrend or reaches S4
            elif close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Determine market regime based on 1d EMA50
            is_uptrend = close[i] > ema50_1d_aligned[i]
            is_downtrend = close[i] < ema50_1d_aligned[i]
            
            # Look for entries with volume confirmation
            # Long entry conditions
            if volume[i] > 1.5 * volume_ma[i]:
                if is_uptrend and close[i] > r4_aligned[i]:
                    # Breakout continuation in uptrend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif not is_uptrend and not is_downtrend and close[i] > s3_aligned[i] and close[i] < r3_aligned[i]:
                    # Reversal from S3 in ranging market
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            # Short entry conditions
            if volume[i] > 1.5 * volume_ma[i]:
                if is_downtrend and close[i] < s4_aligned[i]:
                    # Breakout continuation in downtrend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                elif not is_uptrend and not is_downtrend and close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                    # Reversal from R3 in ranging market
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals