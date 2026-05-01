#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (JAW/TEETH/LIPS) with 1d trend filter and volume confirmation.
# Uses Alligator to identify trendless markets (all lines intertwined) vs trending (lines separated).
# Long: LIPS > TEETH > JAW AND price > 1d EMA50 AND volume > 1.5x 20-bar average.
# Short: LIPS < TEETH < JAW AND price < 1d EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Williams Alligator calculation (6h timeframe)
    # JAW: Blue line - 13-period SMMA smoothed by 8 bars
    # TEETH: Red line - 8-period SMMA smoothed by 5 bars  
    # LIPS: Green line - 5-period SMMA smoothed by 3 bars
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
        result[period-1] = sma[period-1]
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    jaw = smma(jaw, 8)  # SMMA of JAW smoothed by 8
    teeth = smma(close, 8)
    teeth = smma(teeth, 5)  # SMMA of TEETH smoothed by 5
    lips = smma(close, 5)
    lips = smma(lips, 3)  # SMMA of LIPS smoothed by 3
    
    # Alligator conditions: trend when lines are separated
    # All lines intertwined (no trend): JAW ≈ TEETH ≈ LIPS
    # Strong uptrend: LIPS > TEETH > JAW
    # Strong downtrend: LIPS < TEETH < JAW
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Alligator trend conditions
        strong_uptrend = lips_above_teeth[i] and teeth_above_jaw[i]
        strong_downtrend = lips_below_teeth[i] and teeth_below_jaw[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: strong uptrend AND price > 1d EMA50 AND volume confirmation
            if (strong_uptrend and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend AND price < 1d EMA50 AND volume confirmation
            elif (strong_downtrend and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: trend weakens (lines intertwine) OR price < 1d EMA50 (trend change)
            if (not (lips_above_teeth[i] and teeth_above_jaw[i]) or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: trend weakens (lines intertwine) OR price > 1d EMA50 (trend change)
            if (not (lips_below_teeth[i] and teeth_below_jaw[i]) or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals