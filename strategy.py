#!/usr/bin/env python3
"""
Experiment #292: 12h Williams %R Extreme + 1d/1w Regime Filter + Volume Spike
HYPOTHESIS: Williams %R extremes (oversold < -80, overbought > -20) with volume confirmation and 
1d/1w regime filter (ADX < 25 for mean reversion, ADX > 25 for trend) captures reversals in ranging 
markets and continuations in trending markets. Weekly HTF (1w) provides major trend context. 
Discrete sizing (0.25) minimizes fee drag. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_292_12h_williamsr_extreme_1d_1w_regime_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Williams %R and ADX regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # === HTF: 1w data for major trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Williams %R(14) ===
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace(0, np.nan).fillna(method='ffill').values  # handle division by zero
    
    # === 1d Indicators: ADX(14) for regime detection ===
    tr1 = np.maximum(df_1d['high'], df_1d['close'].shift(1)) - np.minimum(df_1d['low'], df_1d['close'].shift(1))
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    dm_plus = pd.Series(dm_plus)
    dm_minus = pd.Series(dm_minus)
    
    di_plus = 100 * (dm_plus.ewm(span=14, min_periods=14).mean() / atr_1d)
    di_minus = 100 * (dm_minus.ewm(span=14, min_periods=14).mean() / atr_1d)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.ewm(span=14, min_periods=14).mean()
    adx_values = adx.fillna(0).values
    
    # === 1w Indicators: EMA(50) for major trend filter ===
    ema_50_1w = df_1w['close'].ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === Align HTF indicators to 12h timeframe ===
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Regime Filter: ADX < 25 = ranging (mean reversion), ADX > 25 = trending (trend follow) ---
        ranging_market = adx_aligned[i] < 25
        trending_market = adx_aligned[i] > 25
        
        # --- Major Trend Filter: 1w EMA(50) ---
        above_weekly_trend = price > ema_50_1w_aligned[i]
        below_weekly_trend = price < ema_50_1w_aligned[i]
        
        # --- Williams %R Extreme Conditions ---
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # --- Exit Logic (ATR-based stoploss and time-based exit) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (calculated from 12h ATR)
                # We'll approximate ATR from price action for simplicity
                atr_approx = np.abs(high[i] - low[i])  # rough ATR proxy
                stop_level = entry_price - 2.5 * atr_approx
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Time exit: max 6 bars (3 days) to prevent overstaying
                if bars_since_entry >= 6:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                atr_approx = np.abs(high[i] - low[i])
                stop_level = entry_price + 2.5 * atr_approx
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if bars_since_entry >= 6:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long in ranging market: Williams %R oversold AND above weekly trend (buy dips in uptrend)
            # OR in trending market: Williams %R oversold AND above weekly trend (strong continuation)
            if ((ranging_market or trending_market) and oversold and above_weekly_trend):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short in ranging market: Williams %R overbought AND below weekly trend (sell rallies in downtrend)
            # OR in trending market: Williams %R overbought AND below weekly trend (strong continuation)
            elif ((ranging_market or trending_market) and overbought and below_weekly_trend):
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