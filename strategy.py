#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ADX trend filter.
# Long when price breaks above Donchian upper band AND volume > 1.8x 20-bar average AND ADX > 25.
# Short when price breaks below Donchian lower band AND volume > 1.8x 20-bar average AND ADX > 25.
# Exit when price crosses the Donchian midpoint (mean of upper and lower band).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull/bear via ADX trend filter (only trade when trending).

name = "4h_Donchian20_VolumeSpike_ADXTrend_v1"
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
    
    # Donchian(20) on primary timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # ADX(14) for trend filter
    # +DM, -DM, TR
    high_diff = high[1:] - high[:-1]
    low_diff = low[:-1] - low[1:]
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Pad to original length
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    tr = np.concatenate([[0.0], tr])
    
    # Smoothed values
    atr_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20, atr_period*2)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(adx_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above upper band, volume confirmation, ADX > 25
            if (curr_high > highest_high[i] and 
                volume_confirm[i] and 
                adx_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band, volume confirmation, ADX > 25
            elif (curr_low < lowest_low[i] and 
                  volume_confirm[i] and 
                  adx_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below midpoint
            if curr_close < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above midpoint
            if curr_close > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals