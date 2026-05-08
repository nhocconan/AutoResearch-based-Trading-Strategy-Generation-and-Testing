# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels on 1-day chart provide strong support/resistance levels. 
# Breakout above R3 (strong resistance) with 1-day uptrend and volume confirmation signals bullish momentum.
# Breakdown below S3 (strong support) with 1-day downtrend and volume confirmation signals bearish momentum.
# Uses 4h timeframe for entries, 1d for trend filter and pivot calculation.
# Target: 20-50 trades/year to minimize fee drag while capturing significant moves.
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formula: 
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # R2 = C + (H-L)*1.1/6
    # R1 = C + (H-L)*1.1/12
    # PP = (H+L+C)/3
    # S1 = C - (H-L)*1.1/12
    # S2 = C - (H-L)*1.1/6
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    
    # Use previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day)
    R3 = np.full_like(close_1d, np.nan)
    S3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        if np.isnan(ph) or np.isnan(pl) or np.isnan(pc):
            continue
            
        rang = ph - pl
        if rang <= 0:
            continue
            
        R3[i] = pc + rang * 1.1 / 4
        S3[i] = pc - rang * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Start from second bar to ensure we have previous day's data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above R3, 1-day uptrend, volume confirmation
            if close_val > r3_val and ema34_1d_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, 1-day downtrend, volume confirmation
            elif close_val < s3_val and ema34_1d_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below R3 or trend turns down
            if close_val < r3_val or ema34_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above S3 or trend turns up
            if close_val > s3_val or ema34_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals