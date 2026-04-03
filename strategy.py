#!/usr/bin/env python3
"""
Experiment #070: 1d Donchian Breakout + 1w Volume Spike + ATR Stoploss

HYPOTHESIS: Daily Donchian(20) breakouts with weekly volume confirmation and ATR-based risk management
will capture sustained trends while minimizing whipsaw. The 1d timeframe targets 7-25 trades/year 
(30-100 total over 4 years) to avoid fee drag. Weekly volume spike (>2x 20-period average) confirms 
institutional participation. ATR(14) stoploss at 2.5x manages risk. Works in both bull (breakout long) 
and bear (breakdown short) markets by trading in direction of weekly trend filter (price > weekly EMA50).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_vol_1w_stoploss_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume spike and trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # Calculate EMA(50) on 1w close for trend filter
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian channels ===
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= donchian_period:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
        donchian_low = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = max(100, donchian_period)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
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
        # Regime filter: Only trade in direction of weekly trend
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: Require volume spike (> 2.0x average)
        volume_spike = vol_ratio_1w_aligned[i] > 2.0
        
        # Long: Price breaks above Donchian high with volume confirmation in weekly uptrend
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            weekly_uptrend
        )
        
        # Short: Price breaks below Donchian low with volume confirmation in weekly downtrend
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            weekly_downtrend
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals