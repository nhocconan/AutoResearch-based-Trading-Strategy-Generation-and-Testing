#!/usr/bin/env python3
# 4h_WilliamsAlligator_ElderRay_Trend
# Hypothesis: 4-hour Williams Alligator identifies trend direction, Elder Ray confirms strength, with volume and ADX regime filter.
# Alligator (SMAs on median price) defines trend: price above all three lines = uptrend, below = downtrend.
# Elder Ray (Bull/Bear Power) measures trend strength vs 13-period EMA.
# Volume confirms breakout strength, ADX > 25 filters for trending markets.
# Designed for 4h to achieve 20-50 trades/year, effective in both bull and bear markets by following strong trends.

name = "4h_WilliamsAlligator_ElderRay_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator: 3 SMAs on median price
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # Elder Ray: Bull Power and Bear Power vs 13-period EMA of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ADX for trend strength (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0
                
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[period:])
        minus_dm_smooth[period-1] = np.mean(minus_dm[period:])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough history for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: price above all three = uptrend, below all = downtrend
        alligator_long = close[i] > jaw[i] and close[i] > teeth[i] and close[i] > lips[i]
        alligator_short = close[i] < jaw[i] and close[i] < teeth[i] and close[i] < lips[i]
        
        # Elder Ray confirmation: strong bull/bear power
        elder_long = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0,i-20):i+1]) * 0.5
        elder_short = bear_power[i] < 0 and abs(bear_power[i]) > np.mean(abs(bear_power[max(0,i-20):i+1])) * 0.5
        
        # ADX filter: trending market
        trending = adx[i] > 25
        
        # Volume confirmation: strong volume
        strong_volume = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: uptrend + bull power + trending + strong volume
            if alligator_long and elder_long and trending and strong_volume:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + bear power + trending + strong volume
            elif alligator_short and elder_short and trending and strong_volume:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakness (price below Alligator teeth or weak bull power)
            if close[i] < teeth[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakness (price above Alligator teeth or weak bear power)
            if close[i] > teeth[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals