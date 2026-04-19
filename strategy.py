#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with 1-day ADX trend filter and volume confirmation.
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips).
# Long when: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 (trending) + volume > 1.5x 20-period average
# Short when: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 (trending) + volume > 1.5x 20-period average
# Exit when: Alligator lines cross (Lips crosses Teeth) or ADX < 20 (trend weakening)
# Williams Alligator identifies trend direction and entry points, ADX filters for trending markets,
# volume confirms breakout strength. Works in both bull (ride trends) and bear (catch reversals).
name = "4h_WilliamsAlligator_ADX25_Volume"
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
    
    # 1-day data for Williams Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three SMMA (Smoothed Moving Average)
    # Jaw: SMMA(close, 13, 8)
    # Teeth: SMMA(close, 8, 5)
    # Lips: SMMA(close, 5, 3)
    def smoothed_moving_average(data, period, shift):
        """Calculate Smoothed Moving Average (SMMA)"""
        sma = np.zeros_like(data)
        sma[:period] = np.nan
        sma[period] = np.mean(data[:period+1])
        for i in range(period+1, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        # Apply shift
        shifted = np.full_like(data, np.nan)
        shifted[shift:] = sma[:-shift] if shift > 0 else sma
        return shifted
    
    jaw_1d = smoothed_moving_average(close_1d, 13, 8)
    teeth_1d = smoothed_moving_average(close_1d, 8, 5)
    lips_1d = smoothed_moving_average(close_1d, 5, 3)
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        """Calculate Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+
        atr = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        # Initial values
        atr[period-1] = np.mean(tr[:period])
        dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
        dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
        minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.mean(dx[period-1:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1D data to 4H timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for Alligator and ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        adx = adx_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 + volume spike
            if (lips > teeth and teeth > jaw and 
                adx > 25 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 + volume spike
            elif (lips < teeth and teeth < jaw and 
                  adx > 25 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Lips crosses below Teeth OR ADX < 20 (trend weakening)
            if lips < teeth or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Lips crosses above Teeth OR ADX < 20 (trend weakening)
            if lips > teeth or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals