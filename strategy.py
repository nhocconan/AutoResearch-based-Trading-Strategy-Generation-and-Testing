#!/usr/bin/env python3
"""
Experiment #1380: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
HYPOTHESIS: Donchian(20) breakouts on 4h capture significant price moves. 
1d EMA50 filter ensures trades align with daily trend (long when price>EMA50, short when price<EMA50). 
Volume confirmation (>1.5x average) filters for institutional participation. 
ATR-based stoploss (2*ATR) manages risk. Designed to work in both bull and bear markets by 
trading breakouts in the direction of the higher timeframe trend. 
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1380_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[high[0] - low[0]], tr])
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_loss = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA50 and Donchian20
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Stoploss Check ---
        if in_position:
            bars_since_entry += 1
            
            # Check stoploss
            if position_side > 0:  # Long
                if price <= stop_loss:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price >= stop_loss:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine 1d trend
            price_vs_ema = price > ema50_1d_aligned[i]
            
            # Long: price breaks above Donchian upper + price > EMA50 (uptrend)
            if price > donchian_upper[i] and price_vs_ema:
                in_position = True
                position_side = 1
                entry_price = price
                stop_loss = price - 2.0 * atr[i]  # 2*ATR stoploss
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price breaks below Donchian lower + price < EMA50 (downtrend)
            elif price < donchian_lower[i] and not price_vs_ema:
                in_position = True
                position_side = -1
                entry_price = price
                stop_loss = price + 2.0 * atr[i]  # 2*ATR stoploss
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals