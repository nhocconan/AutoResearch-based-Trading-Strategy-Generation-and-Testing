#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ADX_Alligator_Combo_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX and Alligator
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Alligator (Williams) on daily
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Shift for future offset (Williams Alligator uses future data)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 6x ATR for volatility filter
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or \
           np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or \
           np.isnan(atr_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr_6h[i]
        
        # ADX > 25 for trending market
        # ADX < 20 for ranging market
        # Alligator: jaws < teeth < lips = downtrend, jaws > teeth > lips = uptrend
        
        adx_val = adx_aligned[i]
        plus_di_val = plus_di_aligned[i]
        minus_di_val = minus_di_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Trend conditions
        is_uptrend = (jaw_val > teeth_val > lips_val) and (adx_val > 25) and (plus_di_val > minus_di_val)
        is_downtrend = (jaw_val < teeth_val < lips_val) and (adx_val > 25) and (minus_di_val > plus_di_val)
        
        # Range conditions
        is_ranging = adx_val < 20
        
        if position == 0:
            # Enter long in uptrend on pullback to lips (Alligator's lips as dynamic support)
            if is_uptrend and price <= lips_val + 0.5 * atr_val:
                signals[i] = 0.25
                position = 1
            # Enter short in downtrend on pullback to lips (dynamic resistance)
            elif is_downtrend and price >= lips_val - 0.5 * atr_val:
                signals[i] = -0.25
                position = -1
            # Mean reversion in ranging market: buy near jaw (support), sell near lips (resistance)
            elif is_ranging:
                if price <= jaw_val + 0.3 * atr_val:  # Near jaw (support)
                    signals[i] = 0.25
                    position = 1
                elif price >= lips_val - 0.3 * atr_val:  # Near lips (resistance)
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: trend reversal or price reaches jaw (Alligator's jaw as target)
            if not is_uptrend or price >= jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or price reaches jaw
            if not is_downtrend or price <= jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals