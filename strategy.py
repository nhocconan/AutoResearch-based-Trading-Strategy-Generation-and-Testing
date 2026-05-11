#!/usr/bin/env python3
name = "6h_ADX_Alligator_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1D data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Elder Ray on 1D: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = ema13_1d - df_1d['low'].values
    
    # Align Elder Ray components
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # ADX on 6H (trend strength)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    atr_period = 14
    atr = np.zeros(n)
    atr[:atr_period] = np.nan
    for i in range(atr_period, n):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=atr_period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=atr_period, adjust=False).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False).mean()
    
    # Williams Alligator on 1D
    sma13 = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().shift(8).values
    sma8 = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().shift(5).values
    sma5 = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw = align_htf_to_ltf(prices, df_1d, sma13)
    teeth = align_htf_to_ltf(prices, df_1d, sma8)
    lips = align_htf_to_ltf(prices, df_1d, sma5)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_align = lips[i] > teeth[i] > jaw[i]
        bearish_align = lips[i] < teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bull power > 0, Bear power < 0, ADX > 25, Bullish Alligator, Volume surge
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] > 0 and 
                adx[i] > 25 and bullish_align and 
                volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull power < 0, Bear power > 0, ADX > 25, Bearish Alligator, Volume surge
            elif (bull_power_aligned[i] < 0 and bear_power_aligned[i] < 0 and 
                  adx[i] > 25 and bearish_align and 
                  volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Elder Ray turns bearish OR Alligator loses alignment OR ADX weakens
            if (bull_power_aligned[i] <= 0 or bear_power_aligned[i] <= 0 or 
                not bullish_align or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Elder Ray turns bullish OR Alligator loses alignment OR ADX weakens
            if (bull_power_aligned[i] >= 0 or bear_power_aligned[i] >= 0 or 
                not bearish_align or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals