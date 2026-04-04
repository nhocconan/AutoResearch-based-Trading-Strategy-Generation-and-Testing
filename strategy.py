#!/usr/bin/env python3
"""
Experiment #2459: 6h Camarilla pivot + volume spike + regime filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with volume confirmation and ADX regime filter captures institutional 
participation at key levels. Works in bull/bear by adapting to volatility regimes.
Discrete sizing (0.25) limits fee drag. Target 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2459_6h_camarilla_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(low).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        
        # Directional Movement
        up_move = pd.Series(high).diff()
        down_move = -pd.Series(low).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
        return adx.values
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    camarilla_r4 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_s4 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        # Camarilla formulas
        camarilla_r4[i] = prev_close + range_val * 1.1 / 2
        camarilla_r3[i] = prev_close + range_val * 1.1 / 4
        camarilla_s3[i] = prev_close - range_val * 1.1 / 4
        camarilla_s4[i] = prev_close - range_val * 1.1 / 2
    
    # Align to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_12h_aligned[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Exit logic
        if in_position:
            if position_side > 0:  # Long
                # Exit conditions
                if is_ranging and price >= camarilla_r3_aligned[i]:  # Take profit at R3 in range
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif is_trending and price >= camarilla_r4_aligned[i]:  # Trail with R4 in trend
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price <= camarilla_s3_aligned[i]:  # Stop loss at S3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit conditions
                if is_ranging and price <= camarilla_s3_aligned[i]:  # Take profit at S3 in range
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif is_trending and price <= camarilla_s4_aligned[i]:  # Trail with S4 in trend
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price >= camarilla_r3_aligned[i]:  # Stop loss at R3
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Ranging market: mean reversion at R3/S3
            if is_ranging:
                # Long at S3 bounce
                if camarilla_s3_aligned[i] - 0.001 * price <= price <= camarilla_s3_aligned[i] + 0.001 * price:
                    in_position = True
                    position_side = 1
                    signals[i] = SIZE
                # Short at R3 rejection
                elif camarilla_r3_aligned[i] - 0.001 * price <= price <= camarilla_r3_aligned[i] + 0.001 * price:
                    in_position = True
                    position_side = -1
                    signals[i] = -SIZE
            # Trending market: breakout continuation at R4/S4
            elif is_trending:
                # Long breakout above R4
                if price > camarilla_r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    signals[i] = SIZE
                # Short breakdown below S4
                elif price < camarilla_s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    signals[i] = -SIZE
    
    return signals