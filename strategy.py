#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Enhanced
Concept: Improved version of the pivot breakout strategy with better risk management and filter tuning.
- Long when price breaks above R1 with volume > 2x average and above daily EMA34
- Short when price breaks below S1 with volume > 2x average and below daily EMA34
- Exit when price returns to previous day's close OR after 48 hours (2 bars) to prevent overtrading
- Uses tighter volume filter (3x average) and adds ADX filter to avoid choppy markets
- Conservative sizing (0.25) to manage drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Enhanced"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily Camarilla pivots ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    # Previous day's close for exit
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # === 12h: EMA34 trend filter from daily ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 12h: ADX filter to avoid choppy markets ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
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
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = 100 * dm_plus14 / np.where(tr14 > 0, tr14, np.nan)
    di_minus = 100 * dm_minus14 / np.where(tr14 > 0, tr14, np.nan)
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) > 0, (di_plus + di_minus), np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0  # Counter to limit trade duration
    
    start_idx = 34  # Ensure enough data for EMA34 and ADX
    
    for i in range(start_idx, n):
        # Get values
        ema34_val = ema34_aligned[i]
        close_val = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        close_barrier_val = prev_close_aligned[i]
        vol_ratio_val = vol_ratio[i]
        adx_val = adx[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema34_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(close_barrier_val) or np.isnan(vol_ratio_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            continue
        
        # Increment trade duration counter
        if position != 0:
            bars_in_trade += 1
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation, above EMA34, and strong trend (ADX > 25)
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 3.0  # Tighter filter
            trend_filter = ema34_val > 0 and close_val > ema34_val
            adx_filter = adx_val > 25
            
            if breakout_long and vol_confirm and trend_filter and adx_filter:
                signals[i] = 0.25
                position = 1
                bars_in_trade = 0
            # Short: Price breaks below S1 with volume confirmation, below EMA34, and strong trend (ADX > 25)
            elif (close_val < s1_val and vol_confirm and 
                  ema34_val > 0 and close_val < ema34_val and adx_filter):
                signals[i] = -0.25
                position = -1
                bars_in_trade = 0
        
        elif position != 0:
            # Exit conditions: price returns to previous day's close OR max 2 bars (24 hours) held
            time_exit = bars_in_trade >= 2
            
            if position == 1:
                price_exit = close_val <= close_barrier_val
                if price_exit or time_exit:
                    signals[i] = 0.0
                    position = 0
                    bars_in_trade = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                price_exit = close_val >= close_barrier_val
                if price_exit or time_exit:
                    signals[i] = 0.0
                    position = 0
                    bars_in_trade = 0
                else:
                    signals[i] = -0.25
    
    return signals