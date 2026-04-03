#!/usr/bin/env python3
"""
Experiment #099: 6h Williams %R + 12h ADX Trend + Volume Confirmation

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe, 
filtered by 12h ADX(14) > 25 to ensure we trade only in trending markets, and 12h volume 
confirmation (> 1.5x average) to ensure institutional participation. In bull markets, we 
buy oversold pullbacks in uptrends; in bear markets, we sell overbought bounces in downtrends. 
ATR-based stoploss manages risk. Targets 12-37 trades/year on 6h timeframe (50-150 total over 
4 years) to minimize fee drag while capturing medium-term swings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter and volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX(14) on 12h
    if len(df_12h) >= 14:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr = np.zeros(len(close_12h))
        tr[0] = high_12h[0] - low_12h[0]
        for i in range(1, len(close_12h)):
            tr[i] = max(high_12h[i] - low_12h[i], 
                       abs(high_12h[i] - close_12h[i-1]), 
                       abs(low_12h[i] - close_12h[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros(len(close_12h))
        dm_minus = np.zeros(len(close_12h))
        for i in range(1, len(close_12h)):
            up_move = high_12h[i] - high_12h[i-1]
            down_move = low_12h[i-1] - low_12h[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+ , DM-
        tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_ma = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_ma = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_ma / tr_ma
        di_minus = 100 * dm_minus_ma / tr_ma
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    else:
        adx_12h_aligned = np.full(n, 0.0)
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Williams %R(14)
    williams_r = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 13)
        highest_high = np.max(high[start_idx:i+1])
        lowest_low = np.min(low[start_idx:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # Neutral when no range
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Require ADX > 25 for trending market ---
        is_trending = adx_12h_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 12h ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.5
        
        # --- Williams %R Conditions ---
        oversold = williams_r[i] < -80  # Oversold
        overbought = williams_r[i] > -20  # Overbought
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit when Williams %R reaches overbought (exit long)
                if williams_r[i] > -20:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit when Williams %R reaches oversold (exit short)
                if williams_r[i] < -80:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R oversold in uptrend with volume confirmation
        long_condition = (
            oversold and 
            is_trending and 
            volume_spike
        )
        
        # Short: Williams %R overbought in downtrend with volume confirmation
        short_condition = (
            overbought and 
            is_trending and 
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals