#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Trend from 1d EMA (50) provides directional bias to avoid counter-trend trades
# 12h Camarilla pivot reversals (buy at S3/S4, sell at R3/R4) capture mean reversion
# Volume > 1.5x average confirms institutional participation at pivot levels
# Works in bull/bear as 1d EMA adapts to trend and pivots adapt to volatility
# Target: 15-30 trades/year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Load 1d data for Camarilla pivots
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivots = []
    for i in range(len(df_1d)):
        if i == 0:
            pivots.append({'S1': np.nan, 'S2': np.nan, 'S3': np.nan, 'S4': np.nan,
                          'R1': np.nan, 'R2': np.nan, 'R3': np.nan, 'R4': np.nan})
        else:
            # Previous day's values
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            range_val = ph - pl
            
            # Camarilla formulas
            s1 = pc - (range_val * 1.0833 / 2)
            s2 = pc - (range_val * 1.1666 / 2)
            s3 = pc - (range_val * 1.2500 / 2)
            s4 = pc - (range_val * 1.5000 / 2)
            r1 = pc + (range_val * 1.0833 / 2)
            r2 = pc + (range_val * 1.1666 / 2)
            r3 = pc + (range_val * 1.2500 / 2)
            r4 = pc + (range_val * 1.5000 / 2)
            
            pivots.append({'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4,
                          'R1': r1, 'R2': r2, 'R3': r3, 'R4': r4})
    
    # Extract pivot levels
    s1 = np.array([p['S1'] for p in pivots])
    s2 = np.array([p['S2'] for p in pivots])
    s3 = np.array([p['S3'] for p in pivots])
    s4 = np.array([p['S4'] for p in pivots])
    r1 = np.array([p['R1'] for p in pivots])
    r2 = np.array([p['R2'] for p in pivots])
    r3 = np.array([p['R3'] for p in pivots])
    r4 = np.array([p['R4'] for p in pivots])
    
    # Align pivot levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long at S3 or S4 with bullish trend and volume
            if (((close[i] <= s3_aligned[i] or close[i] <= s4_aligned[i]) and 
                 above_ema and volume_confirmed) or
                ((close[i] >= s1_aligned[i] and close[i] <= s2_aligned[i]) and 
                 above_ema and volume_confirmed)):
                position = 1
                signals[i] = position_size
            # Enter short at R3 or R4 with bearish trend and volume
            elif (((close[i] >= r3_aligned[i] or close[i] >= r4_aligned[i]) and 
                   below_ema and volume_confirmed) or
                  ((close[i] <= r1_aligned[i] and close[i] >= r2_aligned[i]) and 
                   below_ema and volume_confirmed)):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches opposite pivot or trend changes
            if (close[i] >= r1_aligned[i] or 
                close[i] < ema_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches opposite pivot or trend changes
            if (close[i] <= s1_aligned[i] or 
                close[i] > ema_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Reversal_Volume_v1"
timeframe = "12h"
leverage = 1.0