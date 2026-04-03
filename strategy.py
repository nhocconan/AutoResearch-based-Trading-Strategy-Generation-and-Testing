#!/usr/bin/env python3
"""
Experiment #151: 6h ADX + Williams Alligator Combination

HYPOTHESIS: Williams Alligator identifies trend phases (jaw-teeth-lips alignment) while ADX > 25 confirms trend strength. 
This combination filters out ranging markets and whipsaws. Long when Alligator is bullish (lips > teeth > jaw) and ADX rising; 
short when bearish (lips < teeth < jaw) and ADX rising. Uses 6h timeframe for lower noise and 1d HTF for ADX smoothing. 
Target: 75-150 total trades over 4 years. Works in bull/bear via trend strength filter that adapts to market conditions.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    close = prices["close"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX calculation (smoother trend strength) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original length
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM
        atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Williams Alligator ===
    # Alligator: Jaw (13-period SMMA, 8 offset), Teeth (8-period SMMA, 5 offset), Lips (5-period SMMA, 3 offset)
    def smma(series, period):
        # Smoothed Moving Average: EMA-like but with different smoothing
        return pd.Series(series).ewm(alpha=1/period, adjust=False).mean().values
    
    jaw = smma(high, 13)  # Using high for jaw (typical Alligator uses median price)
    teeth = smma(high, 8)
    lips = smma(high, 5)
    
    # Apply offsets (shift right by offset periods)
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
        
        # --- ADX Trend Strength Filter ---
        strong_trend = adx_1d_aligned[i] > 25.0
        rising_adx = adx_1d_aligned[i] > adx_1d_aligned[i-1] if i > 0 else False
        
        # --- Williams Alligator Alignment ---
        # Bullish: lips > teeth > jaw (all aligned upward)
        bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish: lips < teeth < jaw (all aligned downward)
        bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit when Alligator alignment breaks (trend weakening) OR ADX falls below 20
            alignment_break = not bullish_alligator and not bearish_alligator
            weak_trend = adx_1d_aligned[i] < 20.0
            
            if alignment_break or weak_trend:
                in_position = False
                position_side = 0
            
            if not in_position:
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: bullish Alligator + strong/trending ADX
        if bullish_alligator and strong_trend and rising_adx:
            in_position = True
            position_side = 1
            signals[i] = SIZE
        # Short: bearish Alligator + strong/trending ADX
        elif bearish_alligator and strong_trend and rising_adx:
            in_position = True
            position_side = -1
            signals[i] = -SIZE
    
    return signals