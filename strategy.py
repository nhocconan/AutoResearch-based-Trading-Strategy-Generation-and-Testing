#!/usr/bin/env python3
"""
Experiment #023: 4h Donchian(20) Breakout + 12h Trend + Volume Spike + Choppiness Filter

HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. Combined with 12h trend filter (price > EMA50 for longs, < EMA50 for shorts), 
volume confirmation (>2.0x average), and choppiness regime filter (CHOP < 61.8 for trending markets), this strategy captures explosive moves 
in both bull and bear markets. Uses discrete position sizing (0.30) and ATR-based stoploss. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
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
    
    # === Choppiness Index on primary timeframe (4h) ===
    chop_length = 14
    chop = np.full(n, np.nan)
    
    if n >= chop_length:
        # Calculate True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Sum of TR over chop_length period
        sum_tr = pd.Series(tr).rolling(window=chop_length, min_periods=chop_length).sum().values
        
        # Calculate highest high and lowest low over chop_length period
        highest_high = pd.Series(high).rolling(window=chop_length, min_periods=chop_length).max().values
        lowest_low = pd.Series(low).rolling(window=chop_length, min_periods=chop_length).min().values
        
        # Choppiness Index formula
        denominator = highest_high - lowest_low
        chop = np.where(
            (denominator != 0) & ~np.isnan(sum_tr) & (sum_tr > 0),
            100 * np.log10(sum_tr / denominator) / np.log10(chop_length),
            50.0  # Neutral when undefined
        )
    
    # === Donchian Channel (20) on primary timeframe (4h) ===
    donchian_length = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    if n >= donchian_length:
        upper_channel = pd.Series(high).rolling(window=donchian_length, min_periods=donchian_length).max().values
        lower_channel = pd.Series(low).rolling(window=donchian_length, min_periods=donchian_length).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, donchian_length, chop_length)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 12h EMA50 ---
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Choppiness Filter: Only trade in trending markets (CHOP < 61.8) ---
        trending_market = chop[i] < 61.8
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Donchian Breakout: Long when price breaks above upper channel, Short when breaks below lower channel
        long_condition = (
            close[i] > upper_channel[i] and 
            price_above_12h_ema and 
            volume_spike and 
            trending_market
        )
        
        short_condition = (
            close[i] < lower_channel[i] and 
            price_below_12h_ema and 
            volume_spike and 
            trending_market
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