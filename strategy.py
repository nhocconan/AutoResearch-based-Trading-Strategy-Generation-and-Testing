#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h ADX trend strength + RSI mean reversion + volume confirmation
    # ADX > 25 indicates strong trend (works in bull/bear), RSI < 30/ > 70 for entries
    # Volume spike confirms institutional interest. Trend filter avoids counter-trend trades.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(arr[:period])
                # Wilder's smoothing: prev*(period-1) + current) / period
                for i in range(period, len(arr)):
                    if not np.isnan(result[i-1]):
                        result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_smooth = wilders_smoothing(tr, period)
        dm_plus_smooth = wilders_smoothing(dm_plus, period)
        dm_minus_smooth = wilders_smoothing(dm_minus, period)
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / tr_smooth
        minus_di = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # RSI (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing for average gain/loss
        def wilders_smoothing(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) >= period:
                result[period-1] = np.nanmean(arr[:period])
                for i in range(period, len(arr)):
                    if not np.isnan(result[i-1]):
                        result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        avg_gain = wilders_smoothing(gain, period)
        avg_loss = wilders_smoothing(loss, period)
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume spike (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(30, n):  # Start after indicator warmup
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong trend (ADX > 25) + oversold (RSI < 30) + volume spike
            if adx[i] > 25 and rsi[i] < 30 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend (ADX > 25) + overbought (RSI > 70) + volume spike
            elif adx[i] > 25 and rsi[i] > 70 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Weakening trend (ADX < 20) or RSI mean reversion
            if position == 1:
                if adx[i] < 20 or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if adx[i] < 20 or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_ADX_RSI_Volume_Trend_MeanRev_v1"
timeframe = "4h"
leverage = 1.0