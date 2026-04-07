#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combo with 1d trend filter
# Uses ADX(14) to detect trending markets (ADX > 25) and Williams Alligator
# (Jaw/Teeth/Lips) for entry signals in the direction of trend.
# In ranging markets (ADX < 20), uses mean reversion at Bollinger Bands(20,2).
# 1d EMA50 filter ensures alignment with higher timeframe trend.
# Volume confirmation filters false signals.
# Designed for low frequency (target: 15-30 trades/year) to minimize fee impact.
# Works in bull/bear via regime adaptation: trend following in trending, mean reversion in ranging.

name = "6h_adx_alligator_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator (6h timeframe)
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # SMMA(13,8) - but SMMA is just EMA with alpha=1/period? Actually Williams uses SMMA
    teeth = smma(close, 8)  # SMMA(8,5)
    lips = smma(close, 5)   # SMMA(5,3)
    # Shift jaws/teeth/lips as per Williams (Jaw+8, Teeth+5, Lips+3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # For simplicity, use unshifted but check alignment - common implementation uses the values as is
    
    # ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
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
        
        # Smooth TR, DM+
        tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean().values
        dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * dm_plus_period / tr_period
        minus_di = 100 * dm_minus_period / tr_period
        
        # DX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        
        # ADX
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Bollinger Bands(20,2) for mean reversion in ranging markets
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(adx[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Market regime: trending if ADX > 25, ranging if ADX < 20
        trending = adx[i] > 25
        ranging = adx[i] < 20
        
        # Alligator alignment: all three lines in order
        # Bullish: Lips > Teeth > Jaw
        # Bearish: Lips < Teeth < Jaw
        bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on reverse Alligator alignment or at BB upper
            if not bullish_align or close[i] >= bb_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on reverse Alligator alignment or at BB lower
            if not bearish_align or close[i] <= bb_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            if trending and vol_confirm:
                # Trend following: enter in direction of Alligator and 1d trend
                if bullish_align and uptrend:
                    position = 1
                    signals[i] = 0.25
                elif bearish_align and downtrend:
                    position = -1
                    signals[i] = -0.25
            elif ranging and vol_confirm:
                # Mean reversion: enter at Bollinger Bands extremes
                if close[i] <= bb_lower[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= bb_upper[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals