#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels from daily timeframe (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # shift(1) for previous day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1, S1 (most significant for intraday reversals)
    # R1 = Close + (High - Low) * 1.12
    # S1 = Close - (High - Low) * 1.12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.12
    s1 = prev_close - camarilla_range * 1.12
    
    # Align Camarilla levels to 12h timeframe (wait for daily bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily trend filter: EMA34 on 1d timeframe
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 level + daily uptrend + volume confirmation
            if (price > r1_aligned[i] and 
                price > ema34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S1 level + daily downtrend + volume confirmation
            elif (price < s1_aligned[i] and 
                  price < ema34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to S1 level or daily trend fails
            if (price < s1_aligned[i] or 
                price < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to R1 level or daily trend fails
            if (price > r1_aligned[i] or 
                price > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals