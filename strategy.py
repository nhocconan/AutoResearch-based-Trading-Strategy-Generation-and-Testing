#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter.
# Long when price breaks above Camarilla R3 level AND volume > 2.0x 20-bar average AND CHOP(14) > 61.8 (range regime).
# Short when price breaks below Camarilla S3 level AND volume > 2.0x 20-bar average AND CHOP(14) > 61.8 (range regime).
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels from 1d provide institutional support/resistance. Volume spike confirms breakout strength.
# Chop regime filter ensures we only trade in ranging markets where mean reversion at pivot levels works best.
# Primary timeframe: 12h, HTF: 1d for Camarilla calculation and volume/CHOP.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels, volume, and CHOP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough for CHOP(14) and Camarilla
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous completed 1d bar
    # Camarilla: based on (high, low, close) of previous day
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for previous day to complete)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current 12h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index: CHOP(14) > 61.8 = ranging market (good for mean reversion at pivots)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range14 = max_high14 - min_low14
    
    # Avoid division by zero
    chop_raw = np.where(range14 > 0, 
                        100 * np.log10(np.sum(atr14) / range14) / np.log10(14), 
                        50)  # neutral when no range
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        curr_chop = chop_aligned[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        chop_filter = curr_chop > 61.8  # ranging market regime
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below S3
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND volume confirmation AND chop filter (ranging market)
            if (breakout_up and 
                volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND volume confirmation AND chop filter (ranging market)
            elif (breakout_down and 
                  volume_confirm and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR chop regime ends (trending market)
            if (curr_low < camarilla_s3_aligned[i] or 
                chop_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR chop regime ends (trending market)
            if (curr_high > camarilla_r3_aligned[i] or 
                chop_aligned[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals