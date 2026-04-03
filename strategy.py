#!/usr/bin/env python3
"""
Experiment #131: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation

HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, filtered by 1d weekly pivot levels (using prior week's R1/S1 as trend direction) and 1d volume confirmation, capture medium-term momentum while avoiding false breakouts. Weekly pivot provides structural support/resistance from higher timeframe, volume confirms participation, and ATR-based stoploss manages risk. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot and volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week's OHLC
    if len(df_1d) >= 5:
        # Resample 1d to weekly to get prior week's OHLC
        df_1d_indexed = pd.DataFrame({
            'open': df_1d['open'].values,
            'high': df_1d['high'].values,
            'low': df_1d['low'].values,
            'close': df_1d['close'].values
        }, index=pd.to_datetime(df_1d['open_time']))
        
        # Weekly resample
        df_weekly = df_1d_indexed.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        if len(df_weekly) >= 2:
            # Get prior week's OHLC (shift by 1 to avoid look-ahead)
            prev_week = df_weekly.iloc[-2]  # Second to last row = prior completed week
            ph, pl, pc = prev_week['high'], prev_week['low'], prev_week['close']
            
            # Standard pivot point calculation
            pp = (ph + pl + pc) / 3.0
            r1 = 2 * pp - pl
            s1 = 2 * pp - ph
            r2 = pp + (ph - pl)
            s2 = pp - (ph - pl)
            r3 = r1 + (ph - pl)
            s3 = s1 - (ph - pl)
            
            # Trend direction: above R1 = bullish, below S1 = bearish
            bullish_bias = pc > r1
            bearish_bias = pc < s1
            
            # Create arrays aligned to daily data
            bullish_arr = np.full(len(df_1d), bullish_bias)
            bearish_arr = np.full(len(df_1d), bearish_bias)
            r1_arr = np.full(len(df_1d), r1)
            s1_arr = np.full(len(df_1d), s1)
            
            # Align to 6h timeframe
            bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_arr)
            bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_arr)
            r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_arr)
            s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_arr)
        else:
            bullish_1d_aligned = np.zeros(n, dtype=bool)
            bearish_1d_aligned = np.zeros(n, dtype=bool)
            r1_1d_aligned = np.full(n, np.nan)
            s1_1d_aligned = np.full(n, np.nan)
    else:
        bullish_1d_aligned = np.zeros(n, dtype=bool)
        bearish_1d_aligned = np.zeros(n, dtype=bool)
        r1_1d_aligned = np.full(n, np.nan)
        s1_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Donchian(20) channels - vectorized
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Pivot Trend Filter: Use prior week's R1/S1 for bias ---
        bullish_bias = bullish_1d_aligned[i]
        bearish_bias = bearish_1d_aligned[i]
        
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
                # Take profit at Donchian low (trailing stop)
                if close[i] <= donchian_low[i]:
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
                # Take profit at Donchian high (trailing stop)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and bullish bias from weekly pivot
        long_condition = (
            close[i] > donchian_high[i] and 
            bullish_bias and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian low with volume and bearish bias from weekly pivot
        short_condition = (
            close[i] < donchian_low[i] and 
            bearish_bias and 
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