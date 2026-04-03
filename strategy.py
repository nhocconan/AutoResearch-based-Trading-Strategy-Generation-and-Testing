#!/usr/bin/env python3
"""
Experiment #747: 6h Camarilla Pivot + 1d Volume Spike + ADX Regime Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 1d volume spike (>2.0x average) and ADX regime filter (ADX>25 for trend, 
ADX<20 for range) captures high-probability institutional moves. In ranging markets 
(ADX<20), fade at R3/S3 levels. In trending markets (ADX>25), breakout continuation 
at R4/S4 levels. Uses discrete position sizing (0.25) to minimize fee churn. 
Designed to work in both bull and bear markets by adapting to regime.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_747_6h_camarilla_1d_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #            S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    hl_range = high_1d - low_1d
    camarilla_r4 = close_1d + (hl_range * 1.1 / 2)
    camarilla_r3 = close_1d + (hl_range * 1.1 / 4)
    camarilla_s3 = close_1d - (hl_range * 1.1 / 4)
    camarilla_s4 = close_1d - (hl_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === HTF: 1d data for volume spike detection ===
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(n)
    vol_ratio_1d[20:] = volume[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 6h Indicators: ADX(14) for regime detection ===
    # Calculate True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # Calculate Directional Movement
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_smooth > 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth > 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: ATR(14) for stoploss ===
    atr = tr_smooth  # Already calculated above
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 14)  # sufficient for volume MA and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(adx[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average from 1d)
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        if volume_spike:
            # Regime detection from ADX
            if adx[i] < 20:  # Ranging market - mean reversion at R3/S3
                # Long: price rejects S3 (bounces off support)
                if low[i] <= s3_aligned[i] * 1.001 and close[i] > s3_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price rejects R3 (bounces off resistance)
                elif high[i] >= r3_aligned[i] * 0.999 and close[i] < r3_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif adx[i] > 25:  # Trending market - breakout continuation at R4/S4
                # Long: price breaks above R4 with momentum
                if high[i] > r4_aligned[i] and close[i] > r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price breaks below S4 with momentum
                elif low[i] < s4_aligned[i] and close[i] < s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # Transition regime (20 <= ADX <= 25) - no trade
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

}