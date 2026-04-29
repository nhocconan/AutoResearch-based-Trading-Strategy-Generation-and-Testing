#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA(50) trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3 AND price > 4h EMA(50) AND volume > 2.0x 20-period average
# Short when price breaks below 1h Camarilla S3 AND price < 4h EMA(50) AND volume > 2.0x 20-period average
# Uses 4h/1d for signal direction (trend/regime) and 1h only for entry timing precision.
# Added session filter (08-20 UTC) to reduce noise trades. Discrete position sizing (0.20) to minimize fee drag.
# Timeframe: 1h (primary), HTF: 4h for trend, 1d for regime filter.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dRegime_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to reduce noise
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for choppiness
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_first = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d choppiness index: ATR(14) / (max(high,14) - min(low,14)) * 100
    high_roll_max_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_roll_min_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = high_roll_max_14 - low_roll_min_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop_raw = (atr_1d / chop_denominator) * 100
    chop_1d = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Regime: trending if CHOP < 38.2, ranging if CHOP > 61.8
    # We'll use trending regime only (CHOP < 38.2) for breakout strategy
    trending_regime = chop_1d_aligned < 38.2
    
    # Calculate 1h Camarilla pivots (based on previous day's OHLC)
    # We need to align 1d data to calculate 1h Camarilla levels
    camarilla_coeff = 1.1 / 12  # approximately 0.091666
    
    # Previous day's OHLC for Camarilla calculation
    # Since we're on 1h timeframe, we use the prior 1d bar's OHLC
    camarilla_high = align_htf_to_ltf(prices, df_1d, high_1d)
    camarilla_low = align_htf_to_ltf(prices, df_1d, low_1d)
    camarilla_close = align_htf_to_ltf(prices, df_1d, close_1d)
    
    camarilla_range = camarilla_high - camarilla_low
    camarilla_r3 = camarilla_close + camarilla_range * camarilla_coeff * 3
    camarilla_s3 = camarilla_close - camarilla_range * camarilla_coeff * 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_trending = trending_regime[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price breaks below Camarilla S3
            # 2. Price < 4h EMA(50) (trend fails)
            # 3. Not in trending regime anymore
            if (curr_close < curr_s3 or 
                curr_close < curr_ema or
                not curr_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price breaks above Camarilla R3
            # 2. Price > 4h EMA(50) (trend fails)
            # 3. Not in trending regime anymore
            if (curr_close > curr_r3 or 
                curr_close > curr_ema or
                not curr_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Only trade in trending regime
            if curr_trending:
                # Long entry: price breaks above Camarilla R3 AND price > 4h EMA(50) AND volume spike
                if (curr_close > curr_r3 and 
                    curr_close > curr_ema and 
                    vol_spike):
                    signals[i] = 0.20
                    position = 1
                # Short entry: price breaks below Camarilla S3 AND price < 4h EMA(50) AND volume spike
                elif (curr_close < curr_s3 and 
                      curr_close < curr_ema and 
                      vol_spike):
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals