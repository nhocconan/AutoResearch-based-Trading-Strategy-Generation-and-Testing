#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v3
Concept: Daily pivot point breakout with volume confirmation and trend filter on 12h timeframe.
- Uses daily pivot points (R1, S1) as key support/resistance levels
- Long when price breaks above R1 with volume confirmation and above 12h EMA50
- Short when price breaks below S1 with volume confirmation and below 12h EMA50
- Exit when price returns to central pivot point (mean reversion)
- Conservative sizing (0.20) to manage drawdown and reduce trade frequency
- Added ADX filter to avoid whipsaws in low-trend environments
- Works in bull/bear: Pivot points adapt to market conditions, volume confirms breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Calculate daily pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h: EMA50 trend filter ===
    close = prices['close'].values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 12h: ADX trend strength filter ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di = np.where(tr14 > 0, 100 * plus_dm14 / tr14, 0)
    minus_di = np.where(tr14 > 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50 and ADX
    
    for i in range(start_idx, n):
        # Get values
        ema50_val = ema50[i]
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_val) or np.isnan(pivot_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(adx_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Require ADX > 20 for trend strength
        trend_filter = adx_val > 20
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation, above EMA50, and strong trend
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.5  # Higher threshold to reduce trades
            
            if breakout_long and vol_confirm and close_val > ema50_val and trend_filter:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume confirmation, below EMA50, and strong trend
            elif close_val < s1_val and vol_confirm and close_val < ema50_val and trend_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below central pivot (mean reversion)
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns to or above central pivot (mean reversion)
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals