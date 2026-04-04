#!/usr/bin/env python3
"""
Experiment #5081: 4h Donchian(20) Breakout + 1d/1w HTF Regime + Volume Spike + ATR Stoploss
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts aligned with 1d/1w HTF regime (trending vs ranging) capture strong momentum with controlled frequency. HTF regime filter uses ADX(14) on 1d and 1w timeframes: only take longs when both are trending up (ADX>25 + +DI>-DI) and shorts when both trending down. Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 19-50 trades/year on 4h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts through resistance) and bear markets (breakdowns through support) by requiring HTF trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5081_4h_donchian20_1d_1w_regime_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data for regime filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: ADX(14) for regime ===
    if len(df_1d) >= 14:
        # Calculate True Range
        tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
        tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
        tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Calculate +DM and -DM
        up_move = df_1d['high'].values[1:] - df_1d['high'].values[:-1]
        down_move = df_1d['low'].values[:-1] - df_1d['low'].values[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        tr_smoothed = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm_smoothed = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm_smoothed = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di_1d = 100 * plus_dm_smoothed / tr_smoothed
        minus_di_1d = 100 * minus_dm_smoothed / tr_smoothed
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
        adx_1d = pd.Series(dx_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Regime: trending up when ADX>25 and +DI>-DI
        trending_up_1d = (adx_1d > 25) & (plus_di_1d > minus_di_1d)
        # Regime: trending down when ADX>25 and -DI>+DI
        trending_down_1d = (adx_1d > 25) & (minus_di_1d > plus_di_1d)
        
        # Align to 4h timeframe
        trending_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trending_up_1d.astype(float))
        trending_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trending_down_1d.astype(float))
    else:
        trending_up_1d_aligned = np.full(n, False)
        trending_down_1d_aligned = np.full(n, False)
    
    # === 1w Indicators: ADX(14) for regime ===
    if len(df_1w) >= 14:
        # Calculate True Range
        tr1 = df_1w['high'].values[1:] - df_1w['low'].values[1:]
        tr2 = np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1])
        tr3 = np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1])
        tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Calculate +DM and -DM
        up_move = df_1w['high'].values[1:] - df_1w['high'].values[:-1]
        down_move = df_1w['low'].values[:-1] - df_1w['low'].values[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        tr_smoothed = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm_smoothed = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm_smoothed = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Directional Indicators
        plus_di_1w = 100 * plus_dm_smoothed / tr_smoothed
        minus_di_1w = 100 * minus_dm_smoothed / tr_smoothed
        dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
        adx_1w = pd.Series(dx_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Regime: trending up when ADX>25 and +DI>-DI
        trending_up_1w = (adx_1w > 25) & (plus_di_1w > minus_di_1w)
        # Regime: trending down when ADX>25 and -DI>+DI
        trending_down_1w = (adx_1w > 25) & (minus_di_1w > plus_di_1w)
        
        # Align to 4h timeframe
        trending_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trending_up_1w.astype(float))
        trending_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trending_down_1w.astype(float))
    else:
        trending_up_1w_aligned = np.full(n, False)
        trending_down_1w_aligned = np.full(n, False)
    
    # === 4h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # HTF regime filter: both 1d and 1w must agree on trend direction
        regime_long = trending_up_1d_aligned[i] and trending_up_1w_aligned[i]
        regime_short = trending_down_1d_aligned[i] and trending_down_1w_aligned[i]
        
        # Donchian breakout conditions with HTF regime alignment
        # Long: Donchian breakout above + HTF trending up
        # Short: Donchian breakdown below + HTF trending down
        breakout_long = (price >= high_roll[i]) and regime_long and vol_confirm
        breakout_short = (price <= low_roll[i]) and regime_short and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals