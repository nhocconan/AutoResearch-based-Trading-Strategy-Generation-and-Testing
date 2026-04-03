#!/usr/bin/env python3
"""
Experiment #071: 6h Camarilla pivot breakout + 1d volume spike + ATR stoploss

HYPOTHESIS: Camarilla pivot levels derived from 1d OHLC provide intraday support/resistance 
structure on 6h timeframe. Breakouts above R4 or below S4 with 1d volume confirmation 
indicate institutional participation and continuation. Fades at R3/S3 with volume spike 
capture mean reversion in ranging markets. ATR-based stoploss manages risk. Designed for 
6h timeframe targeting 12-37 trades/year (50-150 total over 4 years) to minimize fee drag 
while working in both bull and bear regimes via dual breakout/fade logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and volume (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    if len(df_1d) >= 1:
        # Previous day OHLC (shifted by 1 for no look-ahead)
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        # Camarilla levels
        range_ = prev_high - prev_low
        camarilla_h5 = prev_close + range_ * 1.1 / 2  # R4
        camarilla_h4 = prev_close + range_ * 1.1 / 4  # R3
        camarilla_h3 = prev_close + range_ * 1.1 / 6  # R2
        camarilla_l3 = prev_close - range_ * 1.1 / 6  # S2
        camarilla_l4 = prev_close - range_ * 1.1 / 4  # S3
        camarilla_l5 = prev_close - range_ * 1.1 / 2  # S4
        
        # Align to 6h timeframe (shift(1) already applied in Camarilla calc)
        h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    else:
        h5_aligned = np.full(n, np.nan)
        h4_aligned = np.full(n, np.nan)
        h3_aligned = np.full(n, np.nan)
        l3_aligned = np.full(n, np.nan)
        l4_aligned = np.full(n, np.nan)
        l5_aligned = np.full(n, np.nan)
    
    # Volume confirmation: 1d volume ratio (current vs 20-period average)
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
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
        if (np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 1d ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
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
                # Take profit at Camarilla H3 (R2) for longs
                if close[i] >= h3_aligned[i]:
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
                # Take profit at Camarilla L3 (S2) for shorts
                if close[i] <= l3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price breaks above H5 (R4) with volume
        long_breakout = (
            close[i] > h5_aligned[i] and 
            volume_spike
        )
        
        # Short breakout: Price breaks below L5 (S4) with volume
        short_breakout = (
            close[i] < l5_aligned[i] and 
            volume_spike
        )
        
        # Long fade: Price rejects at H4 (R3) with volume (mean reversion)
        long_fade = (
            close[i] < h4_aligned[i] and 
            close[i-1] >= h4_aligned[i-1] and  # Was at or above H4
            volume_spike
        )
        
        # Short fade: Price rejects at L4 (S3) with volume (mean reversion)
        short_fade = (
            close[i] > l4_aligned[i] and 
            close[i-1] <= l4_aligned[i-1] and  # Was at or below L4
            volume_spike
        )
        
        if long_breakout or long_fade:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_breakout or short_fade:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals