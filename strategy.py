#!/usr/bin/env python3
"""
Experiment #066: 4h Donchian Breakout + Volume Spike + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture strong momentum moves.
Volume confirmation (>2.0x average volume) filters false breakouts.
ATR-based stoploss (2.5x ATR) manages risk. Works in both bull and bear markets
by trading breakouts in direction of trend. Primary timeframe: 4h, HTF: 1d for trend filter.
Target: 75-200 trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_066_4h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators: Donchian Channels (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian, ATR, EMA stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 2.0  # Volume spike threshold
        
        # --- Trend Filter (1d EMA50) ---
        uptrend = price > ema_1d_aligned[i]
        downtrend = price < ema_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # Check stoploss
            if position_side > 0:  # Long position
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 1 bar
            if bars_since_entry < 1:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: Donchian breakout above upper band in uptrend with volume
        if price > donchian_upper[i-1] and uptrend and vol_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            stop_price = entry_price - 2.5 * atr[i]  # 2.5x ATR stoploss
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Donchian breakout below lower band in downtrend with volume
        elif price < donchian_lower[i-1] and downtrend and vol_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            stop_price = entry_price + 2.5 * atr[i]  # 2.5x ATR stoploss
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals