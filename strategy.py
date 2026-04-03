#!/usr/bin/env python3
"""
Experiment #103: 4h Donchian(20) breakout + 12h trend filter + volume confirmation

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe, filtered by 12h EMA(21) trend and 4h volume 
confirmation, capture medium-term momentum while avoiding false breakouts. The 12h EMA ensures 
alignment with higher timeframe direction, 4h volume confirms participation, and ATR-based 
stoploss manages risk. Targets 19-50 trades/year on 4h timeframe (75-200 total over 4 years) 
to minimize fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        ema_21_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    else:
        ema_21_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian(20) channels
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # Default to 1.0 (neutral)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
        if np.isnan(ema_21_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Price above/below 12h EMA21 ---
        price_above_ema = close[i] > ema_21_12h_aligned[i]
        price_below_ema = close[i] < ema_21_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian low (trailing stop)
                if close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian high (trailing stop)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and 
            price_above_ema and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian low with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and 
            price_below_ema and 
            volume_spike
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