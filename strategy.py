#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_Trend_v1
Hypothesis: On 6h timeframe, combine ADX (>25) for trend strength with Williams Alligator (Jaw/Teeth/Lips) for direction. Long when price > Lips > Teeth > Jaw (bullish alignment), short when price < Lips < Teeth < Jaw (bearish alignment). Uses 1d HTF for higher-timeframe trend filter (price > 1d EMA50 for longs, price < 1d EMA50 for shorts) to avoid counter-trend whipsaws. Designed for low-moderate trade frequency (~15-30/year) to minimize fee drag and work in both bull (trend continuation) and bear (trend continuation) regimes by only trading with strong, aligned trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1-day EMA50 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams Alligator (13,8,5 SMAs shifted) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    median_price = (high + low) / 2  # Williams Alligator uses median price
    
    # Jaw (13-period SMA, shifted by 8 bars)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift right by 8 bars (future)
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth (8-period SMA, shifted by 5 bars)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift right by 5 bars
    teeth[:5] = np.nan
    
    # Lips (5-period SMA, shifted by 3 bars)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift right by 3 bars
    lips[:3] = np.nan
    
    # === ADX (14-period) for trend strength ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_50 = ema_50_1d_aligned[i]
        adx_val = adx[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + bullish Alligator alignment (Lips > Teeth > Jaw) + price > 1d EMA50
            if adx_val > 25 and lips_val > teeth_val and teeth_val > jaw_val and price_close > ema_50:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: ADX > 25 (strong trend) + bearish Alligator alignment (Lips < Teeth < Jaw) + price < 1d EMA50
            elif adx_val > 25 and lips_val < teeth_val and teeth_val < jaw_val and price_close < ema_50:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.5 * ATR from entry (using close-based exit)
            # Recalculate ATR for current bar (simplified - using precomputed would require storing)
            # For simplicity, use fixed percentage stoploss based on entry
            if position == 1:
                if price_close < entry_price * 0.97:  # 3% stoploss
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price * 1.03:  # 3% stoploss
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_Trend_v1"
timeframe = "6h"
leverage = 1.0