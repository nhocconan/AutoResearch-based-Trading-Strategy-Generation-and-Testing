#!/usr/bin/env python3
"""
Experiment #082: 12h Donchian(20) Breakout + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Donchian channel breakouts on 12h timeframe capture medium-term trends in both bull and bear markets.
Volume confirmation from 1d ensures institutional participation, while 1w EMA filter aligns with higher timeframe direction.
This combination reduces false breakouts and focuses on high-probability moves. Targets 12-37 trades/year on 12h timeframe
(50-150 total over 4 years) to minimize fee drag. Uses discrete position sizing (0.25) and ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Calculate Donchian channel (20-period) on 12h
    # We need to get 12h OHLC data
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        # Donchian upper/lower bands (20-period)
        donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        
        # Align to 12h timeframe (already aligned since df_12h is 12h data)
        donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
        donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    else:
        donchian_upper_aligned = np.full(n, np.nan)
        donchian_lower_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in alignment with 1w trend ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using 12h data
            # We need to calculate TR from 12h data
            if len(df_12h) >= 1:
                # Map current index to 12h bar index
                # Since we're on 12h timeframe, each price bar is a 12h bar
                # But we need to calculate ATR using 12h OHLC
                # We'll approximate using current timeframe data for simplicity
                # In practice, we should use 12h data for ATR calculation
                pass
            
            # Calculate ATR(14) on current timeframe (12h) using price data
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
                # Take profit at Donchian lower band (trailing stop for longs)
                if close[i] <= donchian_lower_aligned[i]:
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
                # Take profit at Donchian upper band (trailing stop for shorts)
                if close[i] >= donchian_upper_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume and trend alignment
        long_condition = (
            close[i] > donchian_upper_aligned[i] and 
            volume_spike and 
            price_above_1w_ema
        )
        
        # Short: Price breaks below Donchian lower band with volume and trend alignment
        short_condition = (
            close[i] < donchian_lower_aligned[i] and 
            volume_spike and 
            price_below_1w_ema
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