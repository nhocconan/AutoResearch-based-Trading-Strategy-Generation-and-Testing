#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Point Breakout with 12h Trend Filter and Volume Spike
# - Camarilla pivot levels identify key support/resistance based on prior day's range
# - Breakout above R3 or below S3 with 12h trend alignment (EMA50) captures strong moves
# - Volume spike confirms breakout validity and reduces false signals
# - Works in both bull/bear markets by using trend filter to avoid counter-trend trades
# - Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Formula uses previous day's OHLC: R4, R3, R2, R1, PP, S1, S2, S3, S4
    n1d = len(high_1d)
    R3 = np.full(n1d, np.nan)
    S3 = np.full(n1d, np.nan)
    PP = np.full(n1d, np.nan)
    
    for i in range(1, n1d):  # Start from 1 to use previous day's data
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot Point
        pp = (ph + pl + pc) / 3.0
        # Camarilla levels
        r3 = pc + (ph - pl) * 1.1 / 4.0
        s3 = pc - (ph - pl) * 1.1 / 4.0
        
        PP[i] = pp
        R3[i] = r3
        S3[i] = s3
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed as levels are known at open)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 resistance + 12h uptrend + volume spike
            long_cond = (close[i] > R3_aligned[i] and 
                        ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S3 support + 12h downtrend + volume spike
            short_cond = (close[i] < S3_aligned[i] and 
                         ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below PP (pivot point) or reverse signal with volume
            if close[i] < PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above PP (pivot point) or reverse signal with volume
            if close[i] > PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals