#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R(14) extreme reversal with 1d volume confirmation and ADX trend filter.
Long when Williams %R crosses above -80 (oversold) with volume > 1.5x 20-period 1d average volume AND 1d ADX > 20.
Short when Williams %R crosses below -20 (overbought) with volume > 1.5x 20-period 1d average volume AND 1d ADX > 20.
Exit when Williams %R crosses below -50 for long or above -50 for short.
Uses 4h for primary signals and 1d for volume/ADX confirmation. Designed to catch reversals in both bull and bear markets with low trade frequency.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
"""

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
    
    # Get 1d data for volume MA and ADX
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period volume moving average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Williams %R calculation (14-period) on 4h data
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_14 = calculate_williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or
            np.isnan(wr_14[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period 1d average volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: ADX > 20 indicates sufficient trend strength
        trending = adx_14_aligned[i] > 20
        
        # Williams %R signals
        wr_cross_above_80 = wr_14[i] > -80 and wr_14[i-1] <= -80
        wr_cross_below_20 = wr_14[i] < -20 and wr_14[i-1] >= -20
        wr_cross_below_50 = wr_14[i] < -50 and wr_14[i-1] >= -50
        wr_cross_above_50 = wr_14[i] > -50 and wr_14[i-1] <= -50
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) with volume confirmation and trend
            if (wr_cross_above_80 and volume_confirmed and trending):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) with volume confirmation and trend
            elif (wr_cross_below_20 and volume_confirmed and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -50
            if wr_cross_below_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -50
            if wr_cross_above_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Volume_1dADX20_Reversal"
timeframe = "4h"
leverage = 1.0