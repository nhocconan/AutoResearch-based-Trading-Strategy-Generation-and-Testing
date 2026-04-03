#!/usr/bin/env python3
"""
Experiment #222: 12h Donchian(20) breakout + 1d volume confirmation + chop regime filter
HYPOTHESIS: Uses 12h Donchian channel breakouts for trend capture, filtered by 1d volume spike (>1.5x average) and 1d choppiness regime (CHOP < 38.2 = trending). In trending regimes, we take breakouts in the direction of the trend (determined by 1d EMA50 vs EMA200). In choppy regimes (CHOP >= 38.2), we fade breaks at the Donchian bands (mean reversion). ATR-based stoploss (2*ATR) manages risk. Discrete position sizing (0.25) minimizes fee churn. Target: 75-150 total trades over 4 years (19-37/year). Works in bull markets via trend-following breakouts and in bear markets via mean reversion in chop, with symmetry for longs/shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_222_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume, chop, and trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 and EMA200 for trend direction
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d ATR(14) for chop and stoploss
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr2 = np.abs(np.roll(df_1d['close'].values, 1) - df_1d['close'].values)
    tr = np.maximum(tr1, tr2)
    tr[0] = tr1[0]  # First bar
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 1d Choppiness Index (CHOP) - high = ranging, low = trending
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high) - min(low))) / log10(14)
    sum_atr = pd.Series(atr14_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low + 1e-10)) / np.log10(14)
    chop[np.isnan(chop) | np.isinf(chop)] = 50.0  # Default to neutral
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume MA(20) for spike detection
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h ATR(14) for dynamic stoploss ===
    tr_12h = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
    tr_12h = np.maximum(tr_12h, np.abs(np.roll(close, 1) - close))
    tr_12h[0] = tr_12h[0]
    atr14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # Need enough data for EMA200 on 1d
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr14_12h[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- 1d Regime and Trend Filters ---
        uptrend_1d = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend_1d = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        chop_regime = chop_aligned[i] >= 38.2  # >= 38.2 = ranging/choppy
        trend_regime = chop_aligned[i] < 38.2   # < 38.2 = trending
        
        # --- 1d Volume Confirmation ---
        volume_spike = df_1d['volume'].values[i] > (vol_ma_20_1d_aligned[i] * 1.5)
        
        # --- Donchian Breakout Signals ---
        breakout_up = price > donchian_high[i]
        breakout_down = price < donchian_low[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # Dynamic stop based on 12h ATR
            stop_distance = atr14_12h[i] * 2.0
            
            if position_side > 0:  # Long position
                stop_level = entry_price - stop_distance
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if breakout fails or regime/chop conditions invalidate
                if not (breakout_up or (trend_regime and uptrend_1d) or (chop_regime and not breakout_down)):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + stop_distance
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if breakout fails or regime/chop conditions invalidate
                if not (breakout_down or (trend_regime and downtrend_1d) or (chop_regime and not breakout_up)):
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
        
        # --- New Position Entry Logic (Only if Flat) ---
        if volume_spike:
            # Determine regime and trade accordingly
            if trend_regime:
                # In trending regime: trade breakouts with trend
                if breakout_up and uptrend_1d:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif breakout_down and downtrend_1d:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            else:  # chop_regime
                # In choppy regime: fade breaks (mean reversion)
                if breakout_down:  # Price broke above Donchian high -> expect reversion down
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif breakout_up:  # Price broke below Donchian low -> expect reversion up
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals