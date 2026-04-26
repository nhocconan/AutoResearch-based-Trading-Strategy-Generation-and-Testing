#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Tighten entry conditions from prior version by requiring volume > 2.5x 20-bar MA and adding ATR(14) volatility filter (>0.02) to reduce false breakouts. Keep 12h EMA50 trend filter and Camarilla R3/S3 breakouts. Target 15-30 trades/year (60-120 total over 4 years) to minimize fee drag while maintaining edge in bull/bear markets via 12h trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Previous day's OHLC for Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + (rng * 1.1 / 4)
    camarilla_s3 = close_1d - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (they change only at 1d boundaries)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.5x 20-period average (tighter than 2.0x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    # ATR(14) volatility filter: only trade when volatility > 0.02 (2%)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    volatility_filter = atr > 0.02 * close  # ATR > 2% of price
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 50 for ema, 14 for atr, 1 for camarilla)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        vol_filter = volatility_filter[i]
        
        # Determine 12h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_12h = close_val > ema_50_val
        bearish_12h = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla R3/S3 in trend direction with volume spike and volatility filter
        long_entry = (close_val > camarilla_r3_val) and bullish_12h and vol_spike and vol_filter
        short_entry = (close_val < camarilla_s3_val) and bearish_12h and vol_spike and vol_filter
        
        # Exit conditions: opposite Camarilla level touch (S3 for long, R3 for short)
        exit_long = close_val < camarilla_s3_val
        exit_short = close_val > camarilla_r3_val
        
        # Minimum holding period: 4 bars (to avoid whipsaw)
        min_hold = 4
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0