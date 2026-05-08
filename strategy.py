#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX Trend + Volume + Momentum - Uses ADX(14) to identify strong trends
# in both bull and bear markets, combined with volume confirmation and RSI momentum
# to filter false signals. Targets 25-35 trades/year to minimize fee drag on 4h.
# ADX > 25 indicates trending market (works in both up/down trends), with
# +DI/-DI determining direction, volume spike confirming strength, and RSI
# ensuring momentum alignment.

name = "4h_ADXTrend_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ADX components using Wilder's smoothing
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                for i in range(period, len(data)):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            return result
        
        tr_smoothed = WilderSmoothing(tr, period)
        plus_dm_smoothed = WilderSmoothing(plus_dm, period)
        minus_dm_smoothed = WilderSmoothing(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_smoothed != 0, 100 * plus_dm_smoothed / tr_smoothed, 0)
        minus_di = np.where(tr_smoothed != 0, 100 * minus_dm_smoothed / tr_smoothed, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = WilderSmoothing(dx, period)
        
        return adx, plus_di, minus_di
    
    # Calculate ADX and DI
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # RSI for momentum confirmation
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                for i in range(period, len(data)):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            return result
        
        avg_gain = WilderSmoothing(gain, period)
        avg_loss = WilderSmoothing(loss, period)
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume spike confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending), +DI > -DI (uptrend), RSI > 50 (bullish momentum), volume spike
            long_cond = (adx[i] > 25 and 
                        plus_di[i] > minus_di[i] and
                        rsi[i] > 50 and
                        volume_spike[i])
            
            # Short: ADX > 25 (trending), -DI > +DI (downtrend), RSI < 50 (bearish momentum), volume spike
            short_cond = (adx[i] > 25 and 
                         minus_di[i] > plus_di[i] and
                         rsi[i] < 50 and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX < 20 (trend weakening) OR RSI < 40 (momentum loss)
            if adx[i] < 20 or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (trend weakening) OR RSI > 60 (momentum loss)
            if adx[i] < 20 or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals