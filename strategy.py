#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + ADX Trend Strength + Volume Confirmation
# Long when Jaw < Teeth < Lips (bullish alignment), ADX > 25 (trending), Volume > 1.5x MA(20)
# Short when Jaw > Teeth > Lips (bearish alignment), ADX > 25 (trending), Volume > 1.5x MA(20)
# Williams Alligator identifies trend initiation and direction, ADX filters for strong trends,
# Volume confirms conviction. Weekly trend acts as higher timeframe filter.
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsAlligator_ADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams Alligator (13,8,5) - Smoothed Medians
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set NaN for shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # ADX(14) - Trend Strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: Volume > 1.5x MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        adx_val = adx[i]
        volume_ratio_val = volume_ratio[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = ema34_1w_val > 0
        weekly_downtrend = ema34_1w_val < 0
        
        if position == 0:
            # Enter long: Bullish Alligator alignment, strong trend, volume confirmation, weekly uptrend
            if (jaw_val < teeth_val < lips_val and 
                adx_val > 25 and 
                volume_ratio_val > 1.5 and 
                weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish Alligator alignment, strong trend, volume confirmation, weekly downtrend
            elif (jaw_val > teeth_val > lips_val and 
                  adx_val > 25 and 
                  volume_ratio_val > 1.5 and 
                  weekly_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Any condition breaks
            if not (jaw_val < teeth_val < lips_val) or adx_val < 20 or volume_ratio_val < 1.2 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Any condition breaks
            if not (jaw_val > teeth_val > lips_val) or adx_val < 20 or volume_ratio_val < 1.2 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals