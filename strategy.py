#!/usr/bin/env python3
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
    
    # Load daily data for Donchian and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan, dtype=float)
        
        tr = np.zeros(len(high))
        dm_plus = np.zeros(len(high))
        dm_minus = np.zeros(len(high))
        
        tr[0] = high[0] - low[0]
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0
                
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0
        
        # Smooth TR, DM+
        atr = np.zeros(len(high))
        atr_dm_plus = np.zeros(len(high))
        atr_dm_minus = np.zeros(len(high))
        
        atr[period-1] = np.mean(tr[:period])
        atr_dm_plus[period-1] = np.mean(dm_plus[:period])
        atr_dm_minus[period-1] = np.mean(dm_minus[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            atr_dm_plus[i] = (atr_dm_plus[i-1] * (period-1) + dm_plus[i]) / period
            atr_dm_minus[i] = (atr_dm_minus[i-1] * (period-1) + dm_minus[i]) / period
        
        # Calculate DX
        dx = np.zeros(len(high))
        for i in range(period, len(high)):
            if atr[i] != 0:
                di_plus = 100 * atr_dm_plus[i] / atr[i]
                di_minus = 100 * atr_dm_minus[i] / atr[i]
                if di_plus + di_minus != 0:
                    dx[i] = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                else:
                    dx[i] = 0
            else:
                dx[i] = 0
        
        # Calculate ADX
        adx = np.full(len(high), np.nan)
        if len(dx) >= 2 * period - 1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian Channel (20) on daily
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        if len(high) < period:
            return upper, lower
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        
        return upper, lower
    
    donch_up_1d, donch_low_1d = calculate_donchian(high_1d, low_1d, 20)
    donch_up_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_up_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donch_up_1d_aligned[i]) or
            np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (trending market)
        if adx_1d_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above daily Donchian upper with volume spike in trending market
            if (close[i] > donch_up_1d_aligned[i] and 
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian lower with volume spike in trending market
            elif (close[i] < donch_low_1d_aligned[i] and 
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below daily Donchian lower (trend reversal)
            if close[i] < donch_low_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above daily Donchian upper (trend reversal)
            if close[i] > donch_up_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ADX25_Donchian20_Volume_Breakout"
timeframe = "6h"
leverage = 1.0