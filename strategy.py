#!/usr/bin/env python3
"""
Experiment #325: 12h Camarilla Pivot Breakout + 1d Volume Spike + Choppiness Filter

HYPOTHESIS: 12h Camarilla pivot levels (H3/L3) act as significant support/resistance. 
Breakouts above H3 or below L3 with volume confirmation (>1.8x average) and 
choppiness regime filter (CHOP > 50 = ranging, < 50 = trending) capture strong 
momentum moves. Uses 1d HTF for pivot calculation (more stable than intraday). 
ATR-based stoploss manages risk. Designed for 12h timeframe targeting 12-37 
trades/year (50-150 total over 4 years) to minimize fee drag while capturing 
significant breakout moves in both bull (breakouts with volume) and bear 
(sharp reversals at pivot levels) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_325_12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d bar
    def calculate_camarilla(h, l, c):
        # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
        # We use H3 and L3 for breakout signals
        range_ = h - l
        h3 = c + (range_ * 1.1 / 4)
        l3 = c - (range_ * 1.1 / 4)
        return h3, l3
    
    h3_1d = np.full(len(df_1d), np.nan)
    l3_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        h3_1d[i], l3_1d[i] = calculate_camarilla(
            df_1d['high'].iloc[i], 
            df_1d['low'].iloc[i], 
            df_1d['close'].iloc[i]
        )
    
    # Align HTF levels to LTF (12h) with shift(1) for completed bars only
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high, low, close, period=14):
        """Choppiness Index: higher = ranging, lower = trending"""
        if len(high) < period:
            return np.full_like(high, np.nan)
        
        atr_sum = np.zeros(len(high))
        for i in range(period, len(high)):
            atr_sum[i] = np.sum(np.maximum(high[i-period+1:i+1] - low[i-period+1:i+1],
                                          np.maximum(np.abs(high[i-period+1:i+1] - close[i-period:i]),
                                                   np.abs(low[i-period+1:i+1] - close[i-period:i]))))
        
        # Avoid division by zero
        max_h = np.maximum.accumulate(high)
        min_l = np.minimum.accumulate(low)
        range_max_min = max_h - min_l
        
        chop = np.full(len(high), 50.0)  # Default neutral
        for i in range(period, len(high)):
            if atr_sum[i] > 0 and range_max_min[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / range_max_min[i]) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for ATR and indicator stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Choppiness Index ---
        # CHOP > 50 = ranging market (mean revert), CHOP < 50 = trending (breakout)
        # We want breakouts in trending markets, so CHOP < 50
        trending_regime = chop[i] < 50
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Camarilla Breakout Conditions ---
        breakout_up = close[i] > h3_1d_aligned[i]
        breakout_down = close[i] < l3_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite pivot level reversion (take profit)
                if close[i] < l3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite pivot level reversion (take profit)
                if close[i] > h3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Camarilla breakout up + volume spike + trending regime
        long_condition = breakout_up and volume_spike and trending_regime
        
        # Short: Camarilla breakout down + volume spike + trending regime
        short_condition = breakout_down and volume_spike and trending_regime
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals