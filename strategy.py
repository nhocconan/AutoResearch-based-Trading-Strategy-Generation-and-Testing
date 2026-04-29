#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation
# Uses Williams %R(14) for overbought/oversold extremes (long when < -80, short when > -20)
# 1d ADX > 25 filters for trending markets to avoid false reversals in chop
# Volume > 1.5x 20-period average confirms institutional participation
# Works in bull/bear: ADX filter ensures we only trade strong trends, %R captures exhaustion
# Novelty: Williams %R on 6h with 1d ADX regime filter - under-explored combo for BTC/ETH

name = "6h_WilliamsR_Extreme_1dADX_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14) on 6h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Get 1d data for ADX and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX(14) for trend strength
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = abs(pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1))
    tr3 = abs(pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di_14 = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_14)
    dx_14 = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx_14).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 34)  # warmup for Williams %R, volume MA, and ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(williams_r[i]) or np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        curr_williams_r = williams_r[i]
        curr_adx = adx_14_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and strong trend (ADX > 25)
            if curr_volume_confirm and curr_adx > 25:
                # Bullish entry: Williams %R deeply oversold (< -80)
                if curr_williams_r < -80:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R deeply overbought (> -20)
                elif curr_williams_r > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns from oversold (> -50) or ADX weakens (< 20)
            if curr_williams_r > -50 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns from overbought (< -50) or ADX weakens (< 20)
            if curr_williams_r < -50 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals