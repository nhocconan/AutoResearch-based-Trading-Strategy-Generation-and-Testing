#!/usr/bin/env python3
# 4h_RSI_Trend_Follow_Volume_Filter
# Hypothesis: RSI-based trend following on 4h timeframe with volume confirmation and ADX filter
# Uses RSI(14) > 55 for long entries and < 45 for short entries in trending markets (ADX > 25)
# Volume confirmation ensures institutional participation
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)
# Works in bull/bear via ADX trend filter and RSI momentum

name = "4h_RSI_Trend_Follow_Volume_Filter"
timeframe = "4h"
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
    
    # RSI(14) calculation
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # ADX(14) for trend strength filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Wilder's smoothing
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # Avoid division by zero
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    # Calculate indicators on 4h data
    rsi_4h = calculate_rsi(close, 14)
    adx_4h = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h[i]) or np.isnan(adx_4h[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_4h[i] > 25
        
        if position == 0:
            # Long: RSI > 55 with volume and strong trend
            if (rsi_4h[i] > 55 and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45 with volume and strong trend
            elif (rsi_4h[i] < 45 and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI < 45 or trend weakens (ADX < 20)
            if (rsi_4h[i] < 45) or (adx_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI > 55 or trend weakens (ADX < 20)
            if (rsi_4h[i] > 55) or (adx_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals