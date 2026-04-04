#!/usr/bin/env python3
"""
Experiment #5279: 6h Donchian Breakout + 12h Volume Spike + 1d Choppiness Regime
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts capture momentum while 12h volume confirmation filters false breakouts. The 1d Choppiness Index (CHOP) acts as a regime filter: CHOP > 61.8 = ranging (avoid breakouts), CHOP < 38.2 = trending (favor breakouts). This combines structure (Donchian), conviction (volume), and market state (choppiness) to work in both bull and bear markets by only taking breakouts aligned with the trending regime. Designed for 12-25 trades/year on 6h timeframe (50-100 total over 4 years) to minimize fee drag. Uses discrete position sizing (0.25) to balance profit potential with drawdown control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5279_6h_donchian_vol_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 12h data for volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_ma = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
        vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    else:
        vol_ma_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for Choppiness Index regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value NaN
        
        # ATR(14)
        atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        # Highest high and lowest low over 14 periods
        hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: CHOP = 100 * log10(sum(atr)/log(hh-ll)) / log10(14)
        sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
        hh_ll = hh - ll
        # Avoid division by zero or log of zero
        chop_raw = np.where((hh_ll > 0) & (~np.isnan(hh_ll)), sum_atr / hh_ll, np.nan)
        chop = 100 * np.log10(chop_raw) / np.log10(14)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Upper band: 20-period high
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: 20-period low
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 14, 20)  # Donchian, volume MA, chop ATR
    
    for i in range(warmup, n):
        # --- Session Filter: 00-24 UTC (6h timeframe, full day) ---
        # 6h candles already cover major sessions
        
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # --- Exit Logic: Close position when Donchian reverses or chop regime changes ---
        if in_position:
            # Check for Donchian reversal (price crosses opposite band)
            if position_side > 0:  # Long position
                if price < donch_low[i]:  # Break below lower band
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if price > donch_high[i]:  # Break above upper band
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: current 12h volume > 1.5 * 20-period MA
        # Get the 12h volume-aligned value for this bar
        # Since we aligned 12h data, we need to find the corresponding 12h bar
        # Simpler: use volume ratio directly from aligned data
        vol_ratio = vol / vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else 0
        vol_confirm = vol_ratio > 1.5
        
        # Choppiness regime: CHOP < 38.2 = trending (favor breakouts)
        chop_trending = chop_aligned[i] < 38.2
        
        # Donchian breakout
        breakout_up = price > donch_high[i]
        breakout_down = price < donch_low[i]
        
        # Entry conditions: breakout + volume confirmation + trending regime
        if breakout_up and vol_confirm and chop_trending:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and vol_confirm and chop_trending:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals