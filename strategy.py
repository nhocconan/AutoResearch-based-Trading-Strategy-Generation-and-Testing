#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX regime filter with volume confirmation.
Long when Bull Power > 0 AND ADX > 25 (trending) AND volume > 1.5x 20-period average.
Short when Bear Power < 0 AND ADX > 25 (trending) AND volume > 1.5x 20-period average.
Exit when Elder Power reverses sign OR ADX < 20 (range) OR ATR stoploss hit (2.0).
Uses 1d HTF for ADX regime to avoid whipsaw in sideways markets. Target: 15-30 trades/year.
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
    
    # Get 1d data for ADX regime filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    def calculate_adx(high_arr, low_arr, close_arr, window=14):
        """Average Directional Index"""
        # True Range
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        up_move = high_arr - np.roll(high_arr, 1)
        down_move = np.roll(low_arr, 1) - low_arr
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        tr_smooth = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        return adx
    
    # Calculate ATR (14-period) for stoploss on 6h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    # Get ADX and align to 6h
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR on 6h
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate Elder Ray components on 6h
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 30  # warmup for EMA13 and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma = volume_ma[i]
        atr_val = atr[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + ADX > 25 (trending) + volume spike
            if bull > 0 and adx_val > 25 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Bear Power < 0 (selling pressure) + ADX > 25 (trending) + volume spike
            elif bear < 0 and adx_val > 25 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Bull Power <= 0 (buying pressure gone)
            if bull <= 0:
                exit_signal = True
            
            # Exit 2: ADX < 20 (market ranging)
            elif adx_val < 20:
                exit_signal = True
            
            # Exit 3: ATR-based stoploss (2.0 * ATR below entry)
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Bear Power >= 0 (selling pressure gone)
            if bear >= 0:
                exit_signal = True
            
            # Exit 2: ADX < 20 (market ranging)
            elif adx_val < 20:
                exit_signal = True
            
            # Exit 3: ATR-based stoploss (2.0 * ATR above entry)
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0