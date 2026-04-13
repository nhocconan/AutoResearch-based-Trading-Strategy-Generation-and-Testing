#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h primary timeframe with 1d HTF filter
    # Strategy: Williams Alligator + Elder Ray + ADX regime filter
    # Logic: 
    #   - Use 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5) to determine trend direction
    #   - Use 1d Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for momentum
    #   - Use 6h ADX(14) > 25 to filter for trending markets only
    #   - Long: Alligator bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND ADX > 25
    #   - Short: Alligator bearish (Lips < Teeth < Jaw) AND Bear Power > 0 AND ADX > 25
    #   - Exit: Alligator reverses (Lips crosses Teeth) OR ADX < 20 (trend weakens)
    # Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag
    # Alligator identifies trend, Elder Ray measures momentum, ADX filters choppy markets
    # Works in both bull and bear markets by following the 1d trend with 6h execution
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Alligator and Elder Ray (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 1d data
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price_1d = (high_1d + low_1d) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(median_price_1d, 13)
    teeth_1d = smma(median_price_1d, 8)
    lips_1d = smma(median_price_1d, 5)
    
    # Calculate Elder Ray on 1d data
    # Bull Power = High - EMA13
    # Bear Power = EMA13 - Low
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d
    
    # Calculate ADX on 6h data (primary timeframe) for regime filter
    def calculate_dm(high, low):
        """Calculate Directional Movement"""
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        return np.concatenate([[0.0], plus_dm]), np.concatenate([[0.0], minus_dm])
    
    def calculate_atr(high, low, close, window=14):
        """Calculate Average True Range"""
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr
    
    plus_dm_6h, minus_dm_6h = calculate_dm(high, low)
    tr_6h = calculate_atr(high, low, close, window=14)
    
    # Smooth the DM and TR values
    period_adx = 14
    if len(tr_6h) >= period_adx:
        atr_6h = pd.Series(tr_6h).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
        plus_dm_smooth = pd.Series(plus_dm_6h).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
        minus_dm_smooth = pd.Series(minus_dm_6h).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
        
        # Calculate DI+ and DI-
        plus_di_6h = np.where(atr_6h != 0, (plus_dm_smooth / atr_6h) * 100, 0)
        minus_di_6h = np.where(atr_6h != 0, (minus_dm_smooth / atr_6h) * 100, 0)
        
        # Calculate DX
        dx_6h = np.where((plus_di_6h + minus_di_6h) != 0, 
                        np.abs(plus_di_6h - minus_di_6h) / (plus_di_6h + minus_di_6h) * 100, 0)
        
        # Calculate ADX
        adx_6h = pd.Series(dx_6h).ewm(span=period_adx, adjust=False, min_periods=period_adx).mean().values
    else:
        atr_6h = np.full(len(high), np.nan)
        adx_6h = np.full(len(high), np.nan)
    
    # Align all 1d indicators to 6h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(adx_6h[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions
        alligator_bullish = lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]
        alligator_bearish = lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]
        
        # Elder Ray conditions
        bull_power_positive = bull_power_1d_aligned[i] > 0
        bear_power_positive = bear_power_1d_aligned[i] > 0
        
        # ADX regime filter
        strong_trend = adx_6h[i] > 25
        weak_trend = adx_6h[i] < 20  # exit condition
        
        # Entry conditions
        enter_long = alligator_bullish and bull_power_positive and strong_trend
        enter_short = alligator_bearish and bear_power_positive and strong_trend
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (not alligator_bullish or lips_1d_aligned[i] <= teeth_1d_aligned[i] or weak_trend))
        exit_short = (position == -1 and 
                     (not alligator_bearish or lips_1d_aligned[i] >= teeth_1d_aligned[i] or weak_trend))
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_alligator_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0