#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R3 level AND 1d volume > 1.5x 20-day average AND choppiness index > 61.8 (range regime).
# Short when price breaks below Camarilla S3 level AND 1d volume > 1.5x 20-day average AND choppiness index > 61.8 (range regime).
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla levels from 1d provide intraday structure, volume spike confirms institutional interest,
# choppiness filter ensures we only trade in ranging markets where mean reversion works.
# Primary timeframe: 4h, HTF: 1d for Camarilla levels and volume, 1d for choppiness.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels, volume, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1d: based on previous day's OHLC
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    # Using previous completed day's values (shift 1)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current 1d volume > 1.5x 20-day average
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Choppiness Index: CHOP > 61.8 indicates ranging market (good for mean reversion)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: using 14-period ATR and 14-period high/low range
    tr1 = pd.Series(df_1d['high'].values - df_1d['low'].values)
    tr2 = pd.Series(abs(df_1d['high'].values - df_1d['close'].shift(1).values))
    tr3 = pd.Series(abs(df_1d['low'].values - df_1d['close'].shift(1).values))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (np.log10(14) * (max_high - min_low))) if len(atr) >= 14 else np.full_like(atr, 50.0)
    # Handle edge cases and align
    chop_series = pd.Series(chop, index=df_1d.index)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_series.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        
        if vol_ma_aligned[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (vol_ma_aligned[i] * 1.5)  # Volume spike threshold
        chop_filter = chop_aligned[i] > 61.8  # Range regime: CHOP > 61.8
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND volume confirmation AND range regime
            if (breakout_up and 
                volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND volume confirmation AND range regime
            elif (breakout_down and 
                  volume_confirm and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR chop regime ends
            if (curr_low < camarilla_s3_aligned[i] or 
                chop_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR chop regime ends
            if (curr_high > camarilla_r3_aligned[i] or 
                chop_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals