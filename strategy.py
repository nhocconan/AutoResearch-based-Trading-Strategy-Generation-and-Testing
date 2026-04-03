#!/usr/bin/env python3
"""
Experiment #259: 6h Williams %R + 12h Camarilla Pivot Reversion
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h. 
12h Camarilla pivot levels (H3/L3, H4/L4) provide institutional support/resistance. 
In ranging markets (ADX < 25 on 12h): fade extremes at H3/L3 for mean reversion. 
In trending markets (ADX >= 25 on 12h): breakout continuation at H4/L4 with volume confirmation.
Uses discrete sizing (0.25) and ATR(14) stoploss (2.5x) to manage risk. 
Designed to work in both bull (trend continuation) and bear (mean reversion in ranges) markets.
Target: 75-175 total trades over 4 years (19-44/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_259_6h_williamsr_12h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivots and ADX regime ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous day's OHLC)
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    h3_12h = pivot_12h + range_12h * 1.1 / 2.0  # H3 = Pivot + 1.1*(Range/2)
    l3_12h = pivot_12h - range_12h * 1.1 / 2.0  # L3 = Pivot - 1.1*(Range/2)
    h4_12h = pivot_12h + range_12h * 1.1        # H4 = Pivot + 1.1*Range
    l4_12h = pivot_12h - range_12h * 1.1        # L4 = Pivot - 1.1*Range
    
    # Align 12h Camarilla levels to 6h
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
    # Calculate 12h ADX for regime detection (trending vs ranging)
    # TR calculation
    tr_12h = np.zeros(len(close_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(close_12h)):
        tr_12h[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    atr_12h = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # +DM and -DM
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = np.diff(low_12h, prepend=low_12h[0]) * -1  # invert to positive
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    tr_ema = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_ema = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_ema = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di_12h = 100 * plus_dm_ema / tr_ema
    minus_di_12h = 100 * minus_dm_ema / tr_ema
    
    # DX and ADX
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = np.zeros(n)
    vol_ratio_6h[20:] = volume[20:] / vol_ma_20_6h[20:]
    vol_ratio_6h[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # enough for 14-period indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r_6h[i]) or np.isnan(atr_14_6h[i]) or
            np.isnan(vol_ratio_6h[i]) or np.isnan(h3_12h_aligned[i]) or
            np.isnan(l3_12h_aligned[i]) or np.isnan(h4_12h_aligned[i]) or
            np.isnan(l4_12h_aligned[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.3x average) ---
        volume_spike = vol_ratio_6h[i] > 1.3
        
        # --- Regime Detection: 12h ADX ---
        # ADX < 25 = ranging market (mean revert)
        # ADX >= 25 = trending market (breakout continuation)
        ranging_market = adx_12h_aligned[i] < 25
        trending_market = adx_12h_aligned[i] >= 25
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14_6h[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14_6h[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            if ranging_market:
                # Ranging market: mean reversion at H3/L3
                # Short when price >= H3 and Williams %R > -20 (overbought)
                if price >= h3_12h_aligned[i] and williams_r_6h[i] > -20:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                # Long when price <= L3 and Williams %R < -80 (oversold)
                elif price <= l3_12h_aligned[i] and williams_r_6h[i] < -80:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            else:  # trending_market
                # Trending market: breakout continuation at H4/L4
                # Long when price > H4 and Williams %R rising from oversold
                if price > h4_12h_aligned[i] and williams_r_6h[i] > williams_r_6h[i-1] and williams_r_6h[i] < -50:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short when price < L4 and Williams %R falling from overbought
                elif price < l4_12h_aligned[i] and williams_r_6h[i] < williams_r_6h[i-1] and williams_r_6h[i] > -50:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals