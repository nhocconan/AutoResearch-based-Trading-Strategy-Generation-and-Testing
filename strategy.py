#!/usr/bin/env python3
"""
Experiment #106: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation + chop filter
HYPOTHESIS: 4h Donchian breakouts aligned with daily HMA trend, confirmed by volume spike (>1.5x) and low choppiness (CHOP < 38.2 = trending regime), capture medium-term momentum. Uses discrete sizing (0.25) and ATR stoploss (2.0*ATR). Target: 75-200 total trades over 4 years (19-50/year). Works in bull/bear via trend filter and volatility-based stops. Added choppiness regime filter to reduce whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_106_4h_donchian20_1d_hma_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend and choppiness regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # Calculate HMA(21) on daily close
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = close_1d.ewm(span=half_len, adjust=False).mean()
    wma_full = close_1d.ewm(span=21, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_1d = raw_hma.ewm(span=sqrt_len, adjust=False).mean()
    hma_1d_values = hma_1d.values
    # Trend: 1 if close > HMA, -1 if close < HMA
    daily_trend = np.where(close_1d > hma_1d_values, 1, -1)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # Calculate Choppiness Index (CHOP) on daily timeframe
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log(high/low))) / log10(n)
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(abs(high_1d - close_1d.shift(1)), 
                                  abs(low_1d - close_1d.shift(1))))
    tr_1d.iloc[0] = high_1d.iloc[0] - low_1d.iloc[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean()
    
    chop_period = 14
    sum_atr_14 = atr_1d.rolling(window=chop_period, min_periods=chop_period).sum()
    highest_high_14 = high_1d.rolling(window=chop_period, min_periods=chop_period).max()
    lowest_low_14 = low_1d.rolling(window=chop_period, min_periods=chop_period).min()
    
    chop_raw = 100 * np.log10(sum_atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(chop_period)
    chop_values = chop_raw.fillna(50).values  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(daily_trend_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Regime Filter: Only trade in trending markets (CHOP < 38.2) ---
        trending_regime = chop_aligned[i] < 38.2
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Daily HMA Trend: from 1d data ---
        bullish_trend = daily_trend_aligned[i] > 0
        bearish_trend = daily_trend_aligned[i] < 0
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~32h on 4h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike and trending_regime:
            # Long: breakout above upper channel AND bullish daily trend
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish daily trend
            elif breakout_down and bearish_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals