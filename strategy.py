#!/usr/bin/env python3
"""
Experiment #067: 6h Camarilla Pivot + Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels on 1d timeframe provide high-probability reversal/breakout zones.
At 6h timeframe: fade at R3/S3 levels (mean reversion in range), breakout continuation at R4/S4 levels (trend).
Volume spike confirms institutional participation. Regime filter uses 1w ADX to distinguish trending (ADX>25) 
from ranging (ADX<20) markets, applying appropriate logic. Targets 12-37 trades/year on 6h timeframe 
(50-150 total over 4 years) to minimize fee drag while capturing both reversal and trend opportunities.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_regime_v1"
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
    
    # Calculate Camarilla pivot levels for 1d
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        piv = (high_1d + low_1d + close_1d) / 3
        range_1d = high_1d - low_1d
        
        # Camarilla levels
        r4 = piv + (range_1d * 1.1 / 2)
        r3 = piv + (range_1d * 1.1 / 4)
        s3 = piv - (range_1d * 1.1 / 4)
        s4 = piv - (range_1d * 1.1 / 2)
        
        # Align to 6h timeframe (shifted by 1 for completed bar)
        piv_aligned = align_htf_to_ltf(prices, df_1d, piv)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        piv_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for ADX regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w - low_1w
        tr2 = np.abs(high_1w - np.roll(close_1w, 1))
        tr3 = np.abs(low_1w - np.roll(close_1w, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                           np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                            np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    else:
        adx_aligned = np.full(n, 20.0)  # Default to ranging
    
    # === 6h Indicators ===
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    max_favorable_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(piv_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Trending (ADX>25) vs Ranging (ADX<20) ---
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic ---
        if in_position:
            # Calculate ATR(14) for dynamic stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update max favorable price
                max_favorable_price = max(max_favorable_price, close[i])
                
                # Stoploss: 2.5 * ATR below entry or max favorable
                stop_level = entry_price - 2.5 * atr_14
                trail_stop = max_favorable_price - 1.5 * atr_14
                effective_stop = max(stop_level, trail_stop)
                
                if low[i] < effective_stop:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                
                # Take profit conditions based on regime
                if is_trending:
                    # In trending market: exit at R4 break or reversal signal
                    if close[i] >= r4_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                else:
                    # In ranging market: exit at R3 or mean reversion
                    if close[i] >= r3_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                        
            else:  # Short position
                # Update max favorable price (lowest for shorts)
                max_favorable_price = min(max_favorable_price, close[i])
                
                # Stoploss: 2.5 * ATR above entry or max favorable
                stop_level = entry_price + 2.5 * atr_14
                trail_stop = max_favorable_price + 1.5 * atr_14
                effective_stop = min(stop_level, trail_stop)
                
                if high[i] > effective_stop:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                
                # Take profit conditions based on regime
                if is_trending:
                    # In trending market: exit at S4 break or reversal signal
                    if close[i] <= s4_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                else:
                    # In ranging market: exit at S3 or mean reversion
                    if close[i] <= s3_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Determine entry logic based on regime
        if is_ranging:
            # Ranging market: fade at R3/S3 (mean reversion)
            long_condition = (
                close[i] <= s3_aligned[i] and  # Price at or below S3
                volume_spike and               # Volume confirmation
                close[i] > open[i]             # Bullish candle confirmation
            )
            
            short_condition = (
                close[i] >= r3_aligned[i] and  # Price at or above R3
                volume_spike and               # Volume confirmation
                close[i] < open[i]             # Bearish candle confirmation
            )
        elif is_trending:
            # Trending market: breakout continuation at R4/S4
            long_condition = (
                close[i] >= r4_aligned[i] and  # Price breaks above R4
                volume_spike and               # Volume confirmation
                close[i] > open[i]             # Bullish candle confirmation
            )
            
            short_condition = (
                close[i] <= s4_aligned[i] and  # Price breaks below S4
                volume_spike and               # Volume confirmation
                close[i] < open[i]             # Bearish candle confirmation
            )
        else:
            # Transition regime (ADX between 20-25): no new entries
            long_condition = False
            short_condition = False
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            max_favorable_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            max_favorable_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals