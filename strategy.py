#!/usr/bin/env python3
"""
Experiment #171: 6h Camarilla Pivot + Volume Spike + ADX Regime Filter

HYPOTHESIS: Camarilla pivot levels from 1d timeframe provide institutional support/resistance. 
Breakout above R4 or below S4 with volume confirmation (>1.5x average) and ADX > 25 (trending regime) 
captures strong momentum moves. Fade at R3/S3 with volume exhaustion (<0.8x average) in ranging 
markets (ADX < 20) provides mean-reversion trades. This dual-regime approach adapts to both bull 
and bear markets by switching between breakout and mean-reversion based on ADX. Targets 15-25 
trades/year on 6h timeframe (60-100 total over 4 years) to minimize fee drag while capturing 
high-probability setups.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_171_6h_camarilla_vol_adx_v1"
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
    
    # Calculate Camarilla pivot levels from previous 1d bar
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h6 = np.full(n, np.nan)
    camarilla_l6 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Camarilla formulas
        camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
        camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
        camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
        camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
        camarilla_h6 = prev_close + (prev_high - prev_low) * 1.1 / 6
        camarilla_l6 = prev_close - (prev_high - prev_low) * 1.1 / 6
        
        # Align to 6h timeframe (shifted by 1 for completed bars only)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
        camarilla_l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    else:
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h6_aligned = np.full(n, np.nan)
        camarilla_l6_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ADX(14) for regime detection ===
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        if minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_di_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    dx = np.zeros(n)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike/exhaustion detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Detection: ADX > 25 = trending, ADX < 20 = ranging ---
        is_trending = adx_14[i] > 25
        is_ranging = adx_14[i] < 20
        
        # --- Volume Conditions ---
        volume_spike = vol_ratio[i] > 1.5      # Breakout confirmation
        volume_exhaustion = vol_ratio[i] < 0.8  # Fade confirmation
        
        # --- Price Levels ---
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR for stoploss (using 14-period)
            atr_stop = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values[i]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_stop
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if position_side > 0 and price >= camarilla_h4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_stop
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if position_side < 0 and price <= camarilla_l4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_trending:
            # Trending regime: Breakout trades
            # Long: Break above R4 with volume spike
            long_breakout = (price > camarilla_h4_aligned[i]) and volume_spike
            # Short: Break below S4 with volume spike
            short_breakout = (price < camarilla_l4_aligned[i]) and volume_spike
            
            if long_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif is_ranging:
            # Ranging regime: Mean reversion at H3/L3
            # Long: Pullback to L3 with volume exhaustion (buy the dip)
            long_reversion = (price <= camarilla_l3_aligned[i]) and volume_exhaustion
            # Short: Pullback to H3 with volume exhaustion (sell the rally)
            short_reversion = (price >= camarilla_h3_aligned[i]) and volume_exhaustion
            
            if long_reversion:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_reversion:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Transition regime (ADX between 20-25): stay flat
            signals[i] = 0.0
    
    return signals