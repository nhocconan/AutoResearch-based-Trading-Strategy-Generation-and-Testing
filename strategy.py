#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume + ADX trend filter
# Williams Alligator (13,8,5) for trend direction (Green > Blue > Red = uptrend)
# 1d volume spike (>1.5x average) for conviction
# 12h ADX (14) > 20 to filter trending markets
# Designed to work in trending markets with volume confirmation
# Target: 15-30 trades/year to avoid fee drag
name = "12h_Alligator_ADX_1dVolume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw (Blue): 13-period SMMA smoothed by 8 periods
    # Teeth (Red): 8-period SMMA smoothed by 5 periods
    # Lips (Green): 5-period SMMA smoothed by 3 periods
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_VALUE) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA components
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply smoothing periods
    jaw = smma(jaw_raw, 8)    # Jaw: 13-period smoothed by 8
    teeth = smma(teeth_raw, 5) # Teeth: 8-period smoothed by 5
    lips = smma(lips_raw, 3)   # Lips: 5-period smoothed by 3
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        
        # First ATR is simple average
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
        
        # Subsequent values with smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
        minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 
                      np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])  # First ADX value
        for i in range(2*period-1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x average
        vol_ma = vol_ma_1d_aligned[i]
        volume_filter = vol_ma > 0 and volume[i] > 1.5 * vol_ma
        
        # Trend filter: ADX > 20
        trend_filter = adx[i] > 20
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) + volume + trend
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: Jaw > Teeth > Lips (bearish alignment) + volume + trend
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Lips < Teeth OR ADX < 15 (trend weakening)
            if lips[i] < teeth[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Lips > Teeth OR ADX < 15 (trend weakening)
            if lips[i] > teeth[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals