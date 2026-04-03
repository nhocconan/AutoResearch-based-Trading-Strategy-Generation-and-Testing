#!/usr/bin/env python3
"""
Experiment #341: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by higher timeframe trend (1d EMA50) 
and volume spikes (1d volume ratio), provide robust entry signals in both bull and bear markets. 
The Donchian structure captures institutional breakouts, volume confirms participation, and the 
1d trend filter ensures alignment with higher timeframe direction. Targets 19-50 trades/year 
on 4h timeframe (75-200 total over 4 years) to minimize fee drag while capturing high-probability 
trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter and volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close for trend filter
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # Calculate volume ratio (current vs 20-period average) on 1d for volume confirmation
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate HMA(21) on 4h close for trend confirmation
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        hma = 2 * wma_half - wma_full
        hma_final = pd.Series(hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma_final
    
    hma_21 = calculate_hma(close, 21)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
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
                # Take profit at Donchian Low (trailing exit)
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
                # Take profit at Donchian High (trailing exit)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Update highest/lowest since entry for trailing logic
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price > 1d EMA50 for long, price < 1d EMA50 for short
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # HMA trend: price > HMA for long bias, price < HMA for short bias
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # Long: Donchian breakout above upper band with volume and trend confirmation
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above Donchian high
            volume_spike and                 # Volume confirmation
            price_above_1d_ema and           # Higher timeframe uptrend
            price_above_hma                  # 4h HMA uptrend
        )
        
        # Short: Donchian breakdown below lower band with volume and trend confirmation
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below Donchian low
            volume_spike and                 # Volume confirmation
            price_below_1d_ema and           # Higher timeframe downtrend
            price_below_hma                  # 4h HMA downtrend
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals