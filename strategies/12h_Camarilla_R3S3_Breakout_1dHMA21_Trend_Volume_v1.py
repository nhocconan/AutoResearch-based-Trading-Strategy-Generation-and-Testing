#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d HMA(21) trend + volume spike confirmation
# Uses 12h primary timeframe for Camarilla pivot breakout signals
# 1d HMA(21) confirms medium-term trend direction (avoids counter-trend trades)
# Volume confirmation (2.0x 20-period average) ensures strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla provides clear structure, 1d HMA adds robust trend filter, volume confirms conviction
# Works in both bull and bear markets by only trading in direction of 1d trend

name = "12h_Camarilla_R3S3_Breakout_1dHMA21_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA(21)
    close_1d = pd.Series(df_1d['close'])
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    wma_half = close_1d.rolling(window=half_length, min_periods=half_length).mean()
    wma_full = close_1d.rolling(window=21, min_periods=21).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_1d = raw_hma.rolling(window=sqrt_length, min_periods=sqrt_length).mean().values
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # Camarilla R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We need daily OHLC to compute these levels
    # Since we're on 12h timeframe, we'll use the 1d data to compute levels
    # and align them to 12h bars
    
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_R3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_S3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (1d -> 12h: 2 bars per day)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Camarilla breakout long: price > R3
            # Camarilla breakout short: price < S3
            breakout_long = close[i] > camarilla_R3_aligned[i]
            breakout_short = close[i] < camarilla_S3_aligned[i]
            
            # 1d HMA trend filter: price > HMA for longs, price < HMA for shorts
            hma_long = close[i] > hma_1d_aligned[i]
            hma_short = close[i] < hma_1d_aligned[i]
            
            if breakout_long and hma_long and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif breakout_short and hma_short and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Camarilla breakdown (price < S3) or trend reversal
            if close[i] < camarilla_S3_aligned[i] or close[i] < hma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Camarilla breakout (price > R3) or trend reversal
            if close[i] > camarilla_R3_aligned[i] or close[i] > hma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals