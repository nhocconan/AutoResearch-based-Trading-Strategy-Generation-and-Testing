#!/usr/bin/env python3
"""
Experiment #302: 12h Donchian Breakout + Volume Spike + 1d Trend Filter

HYPOTHESIS: Donchian(20) breakout on 12h timeframe, combined with 12h volume spike confirmation 
and 1d trend filter (price > EMA50), creates a robust breakout strategy that works in both 
bull and bear markets. The Donchian structure provides clear breakout levels, volume confirms 
institutional participation, and the 1d trend filter ensures alignment with higher timeframe 
direction. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize 
fee drag while capturing high-probability breakouts with trend alignment.
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
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for stronger trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Calculate Donchian(20) on 12h
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        vol_12h = df_12h['volume'].values
        
        # Donchian channels
        donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        
        # Volume ratio (current vs 20-period average)
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        
        # Align all 12h indicators to LTF
        donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
        donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
        vol_ratio_12h_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Require alignment with both 1d and 1w EMA50 ---
        bullish_trend = close[i] > ema_50_1d_aligned[i] and close[i] > ema_50_1w_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i] and close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using LTF data
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
                # Take profit at Donchian Low (mean reversion) or extreme extension
                if close[i] <= donchian_low_aligned[i] or close[i] >= donchian_high_aligned[i] * 1.05:
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
                # Take profit at Donchian High (mean reversion) or extreme extension
                if close[i] >= donchian_high_aligned[i] or close[i] <= donchian_low_aligned[i] * 0.95:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian High with volume in bullish trend
        long_condition = (
            close[i] > donchian_high_aligned[i] and 
            volume_spike and 
            bullish_trend
        )
        
        # Short: Break below Donchian Low with volume in bearish trend
        short_condition = (
            close[i] < donchian_low_aligned[i] and 
            volume_spike and 
            bearish_trend
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