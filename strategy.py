#!/usr/bin/env python3
"""
Experiment #032: 12h Camarilla Pivot + Volume Spike + Choppiness Regime Filter

HYPOTHESIS: Camarilla pivot levels on 12h timeframe act as strong support/resistance zones. 
When price touches these levels with volume confirmation (>1.5x average) and the market 
is in a trending regime (Choppiness Index < 38.2), it indicates a high-probability bounce 
or breakout. The 1d trend filter (price > EMA50) ensures alignment with higher timeframe 
trend to avoid counter-trend trades. Targets 12-37 trades/year on 12h timeframe (50-150 
total over 4 years) to minimize fee drag while capturing high-probability mean reversion 
and continuation moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Camarilla Pivot Levels from Previous Day ===
    # Need previous day's high, low, close - we'll use 1d data shifted by 1
    df_1d_for_pivot = get_htf_data(prices, '1d')
    if len(df_1d_for_pivot) >= 2:
        high_1d = df_1d_for_pivot['high'].values
        low_1d = df_1d_for_pivot['low'].values
        close_1d = df_1d_for_pivot['close'].values
        
        # Calculate Camarilla levels for each 1d bar
        camarilla_h3 = np.full(len(close_1d), np.nan)
        camarilla_l3 = np.full(len(close_1d), np.nan)
        camarilla_h4 = np.full(len(close_1d), np.nan)
        camarilla_l4 = np.full(len(close_1d), np.nan)
        
        pivot = (high_1d + low_1d + close_1d) / 3
        range_1d = high_1d - low_1d
        
        camarilla_h3 = pivot + (range_1d * 1.1 / 4)
        camarilla_l3 = pivot - (range_1d * 1.1 / 4)
        camarilla_h4 = pivot + (range_1d * 1.1 / 2)
        camarilla_l4 = pivot - (range_1d * 1.1 / 2)
        
        # Align to 12h timeframe (shift by 1 to avoid look-ahead)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, camarilla_l3)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, camarilla_l4)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
    
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
    
    # === 12h Indicators: Choppiness Index (14) for regime filter ===
    def choppiness_index(high, low, close, window=14):
        """Calculate Choppiness Index"""
        atr_sum = np.zeros_like(close)
        true_range = np.zeros_like(close)
        
        for i in range(len(close)):
            if i == 0:
                true_range[i] = high[i] - low[i]
            else:
                true_range[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            if i < window:
                atr_sum[i] = np.nan
            else:
                atr_sum[i] = np.sum(true_range[i-window+1:i+1])
        
        # Calculate max/min close over window
        max_close = pd.Series(close).rolling(window=window, min_periods=window).max().values
        min_close = pd.Series(close).rolling(window=window, min_periods=window).min().values
        
        # Choppiness formula
        chop = np.full_like(close, np.nan)
        for i in range(window-1, len(close)):
            if atr_sum[i] > 0 and (max_close[i] - min_close[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_close[i] - min_close[i])) / np.log10(window)
            else:
                chop[i] = 50.0  # Neutral when undefined
        
        return chop
    
    chop = choppiness_index(high, low, close, window=14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF EMA50 and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when market is trending (CHOP < 38.2) ---
        trending_regime = chop[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Trend Filter: Only trade in direction of 1d EMA50 ---
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Camarilla Level Touch Conditions ---
        # L3 and H3 are mean reversion levels (buy at L3, sell at H3)
        # H4 and L4 are breakout levels (buy above H4, sell below L4)
        touch_l3 = low[i] <= camarilla_l3_aligned[i] * 1.001  # Allow small buffer
        touch_h3 = high[i] >= camarilla_h3_aligned[i] * 0.999
        breakout_h4 = high[i] > camarilla_h4_aligned[i]
        breakout_l4 = low[i] < camarilla_l4_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if position_side == 1 and high[i] >= camarilla_h3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if position_side == -1 and low[i] <= camarilla_l3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Touch L3 (mean reversion) OR break above H4 (continuation) + volume spike + trending regime + price above EMA50
        long_condition = ((touch_l3 or breakout_h4) and 
                         volume_spike and 
                         trending_regime and 
                         price_above_ema)
        
        # Short: Touch H3 (mean reversion) OR break below L4 (continuation) + volume spike + trending regime + price below EMA50
        short_condition = ((touch_h3 or breakout_l4) and 
                          volume_spike and 
                          trending_regime and 
                          price_below_ema)
        
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