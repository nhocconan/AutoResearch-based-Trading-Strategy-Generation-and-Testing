#!/usr/bin/env python3
"""
Experiment #012: 12h Donchian(20) Breakout + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Donchian channel breakouts on 12h timeframe capture significant momentum moves. 
Combined with 1d volume confirmation (>2.0x 20-period average) and 1week trend filter (price > EMA50 for longs, < EMA50 for shorts), 
this strategy enters strong trends with institutional participation. Target: 12-37 trades/year on 12h (50-150 total over 4 years) 
to minimize fee drag. Uses discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) for risk management.
Works in both bull (breakouts capture momentum) and bear (shorts on downside breakouts) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
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
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for Donchian(20) channels (Call ONCE before loop) ===
    df_1d_donchian = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) high and low from 1d data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if len(df_1d_donchian) >= 20:
        high_1d = df_1d_donchian['high'].values
        low_1d = df_1d_donchian['low'].values
        
        # Calculate 20-period rolling high and low
        high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
        
        # Align to 12h timeframe (shift by 1 to avoid look-ahead)
        donchian_high_aligned = align_htf_to_ltf(prices, df_1d_donchian, high_ma_20)
        donchian_low_aligned = align_htf_to_ltf(prices, df_1d_donchian, low_ma_20)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
    
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
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 1week EMA50 ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian(20) high with volume confirmation and uptrend
        long_condition = (
            close[i] > donchian_high_aligned[i] and 
            volume_spike and 
            price_above_1w_ema
        )
        
        # Short: Price breaks below Donchian(20) low with volume confirmation and downtrend
        short_condition = (
            close[i] < donchian_low_aligned[i] and 
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