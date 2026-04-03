#!/usr/bin/env python3
"""
Experiment #253: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 4h capture significant price movements with institutional participation. 
Filtering by 12h HMA(21) trend alignment ensures trades follow the higher timeframe momentum, reducing whipsaw. 
Volume confirmation (>1.5x 20-period average) validates breakout strength. ATR-based stoploss (2x) manages risk. 
Discrete position sizing (0.25) minimizes fee churn. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) 
to balance statistical significance with fee drag minimization. Works in both bull and bear markets by capturing 
breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_12h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    else:
        hma_21_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    entry_atr = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * entry_atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * entry_atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Using previous bar's Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Using previous bar's Donchian low
        
        # Trend filter: price relative to 12h HMA
        price_above_hma = close[i] > hma_21_12h_aligned[i]
        price_below_hma = close[i] < hma_21_12h_aligned[i]
        
        # Long: Donchian breakout up + price above 12h HMA + volume confirmation
        if breakout_up and price_above_hma and volume_confirmed:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr_14[i]
            signals[i] = SIZE
        # Short: Donchian breakout down + price below 12h HMA + volume confirmation
        elif breakout_down and price_below_hma and volume_confirmed:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr_14[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals