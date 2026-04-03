#!/usr/bin/env python3
"""
Experiment #391: 6h Elder Ray + 1d ADX Regime + Volume Spike

HYPOTHESIS: Elder Ray (Bull/Bear Power) measures trend strength via EMA13 deviation.
Combined with 1d ADX regime filter (ADX>25 = trending) and volume confirmation (>1.5x average),
this strategy captures strong momentum moves in both bull and bear markets while avoiding
choppy/range-bound conditions. Elder Ray works in bull markets (positive Bull Power) and
bear markets (negative Bear Power). Higher timeframe (1d) regime filter reduces whipsaw.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1d_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d
    if len(df_1d) >= 30:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr = np.maximum(high_1d - low_1d, 
                        np.maximum(abs(high_1d - np.roll(close_1d, 1)), 
                                   abs(low_1d - np.roll(close_1d, 1))))
        tr[0] = high_1d[0] - low_1d[0]
        
        # Directional Movement
        dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                           np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                            np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
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
        
        # Regime: ADX > 25 = trending market
        adx_regime = (adx > 25).astype(np.float64)
        adx_regime_aligned = align_htf_to_ltf(prices, df_1d, adx_regime)
    else:
        adx_regime_aligned = np.zeros(n)  # No trend regime if insufficient data
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Indicators: Calculate EMA13 for Elder Ray ===
    if n >= 13:
        ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
        # Bull Power = High - EMA13
        bull_power = high - ema13
        # Bear Power = Low - EMA13
        bear_power = low - ema13
    else:
        ema13 = np.full(n, np.nan)
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # === Session filter: 00-23 UTC (trade all hours for 6h timeframe) ===
    hours = prices.index.hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # --- Session Filter: Trade all hours for 6h timeframe ---
        hour = hours[i]
        
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_regime_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
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
                # Take profit when Bull Power turns negative
                if bull_power[i] <= 0:
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
                # Take profit when Bear Power turns positive
                if bear_power[i] >= 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bull Power > 0 (strong buying pressure) + trending regime + volume confirmation
        long_condition = (
            bull_power[i] > 0 and  # Buying pressure > 0
            adx_regime_aligned[i] > 0.5 and  # Trending regime (ADX > 25)
            vol_ratio_1d_aligned[i] > 1.5  # Volume confirmation
        )
        
        # Short: Bear Power < 0 (strong selling pressure) + trending regime + volume confirmation
        short_condition = (
            bear_power[i] < 0 and  # Selling pressure < 0
            adx_regime_aligned[i] > 0.5 and  # Trending regime (ADX > 25)
            vol_ratio_1d_aligned[i] > 1.5  # Volume confirmation
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