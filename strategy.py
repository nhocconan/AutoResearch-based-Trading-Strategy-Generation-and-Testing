#!/usr/bin/env python3
"""
6h_Pivot_R3_S3_Fade_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, price tends to revert from daily Camarilla R3/S3 levels during ranging markets (ADX<25), but breaks through R4/S4 during strong trends (ADX>25). 
In ranging markets: fade R3/S3 with volume confirmation. 
In trending markets: breakout continuation at R4/S4 with volume confirmation and 1d trend filter.
Uses discrete sizing (0.25) to limit fee churn. Designed for BTC/ETH in both bull/bear regimes via ADX regime filter.
Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivot and ADX trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    r3_1d = df_1d_close + 0.55 * range_1d
    s3_1d = df_1d_close - 0.55 * range_1d
    r4_1d = df_1d_close + 0.825 * range_1d
    s4_1d = df_1d_close - 0.825 * range_1d
    
    # Align 1d Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1d ADX (14-period) for regime filter ===
    high_1d = df_1d_high
    low_1d = df_1d_low
    close_1d = df_1d_close
    
    # Calculate True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smooth TR and DM
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) 
            or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])
            or np.isnan(adx_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        adx = adx_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Regime filter: ADX < 25 = ranging (fade R3/S3), ADX > 25 = trending (breakout R4/S4)
            if adx < 25:
                # Ranging market: fade at R3/S3 with volume confirmation
                long_condition = (price < r3) and (price > s3) and volume_confirmed
                short_condition = (price < r3) and (price > s3) and volume_confirmed
                
                # Mean reversion: long near S3, short near R3
                if price < (s3 + (r3 - s3) * 0.3):  # Near S3 (30% of range from S3)
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price > (r3 - (r3 - s3) * 0.3):  # Near R3 (30% of range from R3)
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:
                # Trending market: breakout continuation at R4/S4 with volume confirmation
                long_condition = (price > r4) and volume_confirmed
                short_condition = (price < s4) and volume_confirmed
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (ADX drops below 20)
            elif adx < 20:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at midpoint in ranging market
            elif adx < 25 and price > (r3 + s3) / 2:
                signals[i] = 0.0
                position = 0
            # Profit taking at extreme levels
            elif adx >= 25 and price > r4 + (r4 - s4) * 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (ADX drops below 20)
            elif adx < 20:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at midpoint in ranging market
            elif adx < 25 and price < (r3 + s3) / 2:
                signals[i] = 0.0
                position = 0
            # Profit taking at extreme levels
            elif adx >= 25 and price < s4 - (r4 - s4) * 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R3_S3_Fade_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0