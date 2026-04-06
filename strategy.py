#!/usr/bin/env python3
"""
6h ADX + Volume + 1d Parabolic SAR
Hypothesis: ADX > 25 identifies strong trends, Parabolic SAR provides entry/exit signals with trend direction.
Volume confirms momentum. Works in bull (ADX up + SAR below price) and bear (ADX up + SAR above price).
Target: 80-150 total trades over 4 years (20-37/year). Uses discrete position sizing to minimize churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14411_6h_adx_psar_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Parabolic SAR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize PSAR arrays
    psar = np.zeros(len(close_1d))
    psar_long = True
    af = af_start
    ep = high_1d[0]  # extreme point
    psar[0] = low_1d[0]
    
    # Calculate PSAR
    for i in range(1, len(close_1d)):
        if psar_long:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Reverse if price drops below SAR
            if low_1d[i] < psar[i]:
                psar_long = False
                psar[i] = ep  # SAR becomes previous high
                af = af_start
                ep = low_1d[i]  # new extreme point
            else:
                if high_1d[i] > ep:
                    ep = high_1d[i]
                    af = min(af + af_increment, af_max)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Reverse if price rises above SAR
            if high_1d[i] > psar[i]:
                psar_long = True
                psar[i] = ep  # SAR becomes previous low
                af = af_start
                ep = high_1d[i]  # new extreme point
            else:
                if low_1d[i] < ep:
                    ep = low_1d[i]
                    af = min(af + af_increment, af_max)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 6h timeframe
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)  # Require at least 70% of average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 30  # enough for ADX and volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(psar_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction from ADX and DI
        strong_trend = adx_aligned[i] > 25
        bullish = plus_di_aligned[i] > minus_di_aligned[i]
        bearish = plus_di_aligned[i] < minus_di_aligned[i]
        
        # PSAR signals
        psar_bullish = close[i] > psar_aligned[i]  # price above SAR
        psar_bearish = close[i] < psar_aligned[i]  # price below SAR
        
        # Check exits
        if position == 1:  # long position
            # Exit: trend weakens OR SAR flips bearish OR stoploss
            if (not strong_trend or not bullish or not psar_bullish or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend weakens OR SAR flips bullish OR stoploss
            if (not strong_trend or not bearish or not psar_bearish or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: strong trend + PSAR alignment + volume
            long_setup = strong_trend and bullish and psar_bullish and vol_filter[i]
            short_setup = strong_trend and bearish and psar_bearish and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals