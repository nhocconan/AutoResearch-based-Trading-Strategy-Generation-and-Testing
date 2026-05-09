#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and ADX trend filter.
# Works in bull (breakouts continue) and bear (mean-reversion after volatility spikes).
# Uses discrete position sizing to minimize fee churn. Target: 20-50 trades/year.
name = "4h_Donchian20_1dVolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume spike: volume > 2.0 * 20-period SMA of volume
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_sma20
    
    # 4h ADX trend filter (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        plus_dm_smoothed = np.zeros_like(high)
        minus_dm_smoothed = np.zeros_like(high)
        
        plus_dm_smoothed[period-1] = np.sum(plus_dm[:period])
        minus_dm_smoothed[period-1] = np.sum(minus_dm[:period])
        
        for i in range(period, len(high)):
            plus_dm_smoothed[i] = plus_dm_smoothed[i-1] - (plus_dm_smoothed[i-1] / period) + plus_dm[i]
            minus_dm_smoothed[i] = minus_dm_smoothed[i-1] - (minus_dm_smoothed[i-1] / period) + minus_dm[i]
        
        plus_di = 100 * plus_dm_smoothed / atr
        minus_di = 100 * minus_dm_smoothed / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(high)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    adx_trend = adx > 25  # strong trend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(vol_sma20[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Donchian breakout + volume spike + ADX trend
            if (price > highest_high[i] and
                vol_spike[i] and
                adx_trend[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: Donchian breakdown + volume spike + ADX trend
            elif (price < lowest_low[i] and
                  vol_spike[i] and
                  adx_trend[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit: Donchian mean reversion or trend weakness
            if (price < lowest_low[i] or  # re-entry of breakdown level
                adx[i] < 20):             # trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Donchian mean reversion or trend weakness
            if (price > highest_high[i] or  # re-entry of breakout level
                adx[i] < 20):               # trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals