#!/usr/bin/env python3
"""
Experiment #3339: 6h Donchian Breakout + 12h ADX Trend + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts with 12h ADX(14) > 25 trend filter and volume confirmation (>2.0x 20-period average) capture strong trends while avoiding chop. 
ADX ensures we only trade in trending regimes, reducing whipsaws in sideways markets. Works in bull (breakouts with momentum) and bear (strong downtrend continuations) by filtering for actual trend strength.
Target: 75-150 total trades over 4 years (19-37/year). Position size: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3339_6h_donchian20_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        # Smoothed values
        tr_period = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        plus_dm_period = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
        minus_dm_period = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
        # Directional Indicators
        plus_di = 100 * plus_dm_period / tr_period
        minus_di = 100 * minus_dm_period / tr_period
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, lookback, 20, 14, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss at 2.5*ATR or mean reversion to channel midpoint ---
        if in_position:
            if position_side > 0:  # Long
                # Stoploss: 2.5*ATR below entry
                if price < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Mean reversion exit: price returns to Donchian midpoint
                elif price <= (highest_high[i] + lowest_low[i]) / 2:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Stoploss: 2.5*ATR above entry
                if price > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Mean reversion exit: price returns to Donchian midpoint
                elif price >= (highest_high[i] + lowest_low[i]) / 2:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require: volume spike (> 2.0x average) AND strong trend (ADX > 25)
        volume_spike = vol_ratio[i] > 2.0
        strong_trend = adx_12h_aligned[i] > 25
        
        if volume_spike and strong_trend:
            # Long entry: price breaks above Donchian high
            if price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low
            elif price < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>