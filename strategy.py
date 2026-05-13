#!/usr/bin/env python3
name = "12h_TRIX_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX (15-period EMA of EMA of EMA of price change)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    pct_change = ema3.pct_change() * 100
    trix = pct_change.ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > 2.0 * vol_ma_20
    
    # Daily trend filter (1d EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Chop filter: Chop > 61.8 = range (mean reversion), Chop < 38.2 = trending (trend follow)
    # Using 14-period Chop on 12h data
    atr_period = 14
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / np.nansum(tr, axis=0)) / np.log10(atr_period)
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if np.isnan(trix[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend regime: Chop < 38.2 = trending (use TRIX momentum)
        # Range regime: Chop >= 38.2 = range (mean reversion at extremes)
        trending = chop[i] < 38.2
        
        if position == 0:
            if trending:
                # TRENDING: TRIX momentum with volume spike
                if trix[i] > 0 and trix[i-1] <= 0 and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif trix[i] < 0 and trix[i-1] >= 0 and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # RANGE: Mean reversion at extremes
                if trix[i] < -0.1 and vol_spike[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif trix[i] > 0.1 and vol_spike[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
            if position == 0:
                signals[i] = 0.0
        elif position == 1:
            # EXIT: TRIX crosses zero or chop shifts to extreme range
            if trix[i] < 0 or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT: TRIX crosses zero or chop shifts to extreme range
            if trix[i] > 0 or chop[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals