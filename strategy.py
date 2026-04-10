#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and 1w trend filter
# - Entry: Long when price breaks above S4 level + 1d volume > 2.0x 20-period average + 1w close > 1w open (bullish weekly candle)
#          Short when price breaks below R4 level + 1d volume > 2.0x 20-period average + 1w close < 1w open (bearish weekly candle)
# - Exit: Close-based reversal - exit long when price < S3, exit short when price > R3
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Camarilla pivots from 1d for structure, volume for confirmation, 1w for trend filter
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1d_1w_camarilla_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d OHLC for Camarilla pivots and volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 1w OHLC for trend filter
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels from 1d
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # S1 = Close - Range * 1.1 / 12
    # S2 = Close - Range * 1.1 / 6
    # S3 = Close - Range * 1.1 / 4
    # S4 = Close - Range * 1.1 / 2
    # R1 = Close + Range * 1.1 / 12
    # R2 = Close + Range * 1.1 / 6
    # R3 = Close + Range * 1.1 / 4
    # R4 = Close + Range * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    s2_1d = close_1d - range_1d * 1.1 / 6.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    r2_1d = close_1d + range_1d * 1.1 / 6.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w bullish/bearish candle
    bullish_week = close_1w > open_1w  # True for bullish weekly candle
    bearish_week = close_1w < open_1w  # True for bearish weekly candle
    
    # Align all HTF data to 6h timeframe
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    bullish_week_aligned = align_htf_to_ltf(prices, df_1w, bullish_week.astype(float))
    bearish_week_aligned = align_htf_to_ltf(prices, df_1w, bearish_week.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(s4_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(bullish_week_aligned[i]) or 
            np.isnan(bearish_week_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close
        close_price = close_6h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above S4 + volume confirmation + bullish weekly candle
            if (close_price > s4_1d_aligned[i] and 
                volume_confirmation and 
                bullish_week_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below R4 + volume confirmation + bearish weekly candle
            elif (close_price < r4_1d_aligned[i] and 
                  volume_confirmation and 
                  bearish_week_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < S3
            # Exit short when price > R3
            if position == 1:
                if close_price < s3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > r3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals