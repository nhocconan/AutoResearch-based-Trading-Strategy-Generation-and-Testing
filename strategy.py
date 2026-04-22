#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Squeeze breakout with 1-day volume confirmation and ADX trend filter.
Long when Bollinger Bands width at 50-day low, price breaks above upper band, 1-day volume > 1.5x 20-day average, and ADX > 25.
Short when Bollinger Bands width at 50-day low, price breaks below lower band, 1-day volume > 1.5x 20-day average, and ADX > 25.
Exit when price crosses the middle band (20-day SMA).
Bollinger Squeeze identifies low volatility breakouts; volume confirmation ensures institutional participation; ADX filter avoids choppy markets.
Designed for low trade frequency by requiring multiple confirmations (volatility contraction + volume expansion + trend strength).
Works in both bull and bear markets by capturing volatility breakouts regardless of direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for volume and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1-day volume: 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 1-day ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        dx[2*period:] = 100 * np.abs(plus_di[2*period:] - minus_di[2*period:]) / (plus_di[2*period:] + minus_di[2*period:])
        
        adx[2*period] = np.mean(dx[2*period:3*period+1])
        for i in range(3*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Bollinger Bands (20, 2) on 4h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20
    
    # Bollinger Squeeze: BB width at 50-day low
    bb_width_50d_low = pd.Series(bb_width).rolling(window=50, min_periods=50).min().values
    squeeze_condition = bb_width <= bb_width_50d_low * 1.1  # Within 10% of 50-day low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for BB width 50-day low
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger squeeze, price breaks above upper band, volume confirmation, ADX > 25
            if (squeeze_condition[i] and 
                close[i] > upper_band[i] and 
                volume[i] > 1.5 * vol_ma_20_aligned[i] and 
                adx_14_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze, price breaks below lower band, volume confirmation, ADX > 25
            elif (squeeze_condition[i] and 
                  close[i] < lower_band[i] and 
                  volume[i] > 1.5 * vol_ma_20_aligned[i] and 
                  adx_14_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses middle band (20-day SMA)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below middle band
                if close[i] < sma_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above middle band
                if close[i] > sma_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Bollinger_Squeeze_1dVolume_ADX_Filter"
timeframe = "4h"
leverage = 1.0