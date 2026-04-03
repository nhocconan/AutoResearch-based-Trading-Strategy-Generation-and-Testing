#!/usr/bin/env python3
"""
Experiment #263: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Combining 4h Donchian channel breakouts with 4h HMA trend alignment and volume confirmation creates a robust trend-following strategy that works in both bull and bear markets. The Donchian(20) captures price channel breakouts, the HMA(21) filters for trend direction, and volume confirmation ensures institutional participation. Uses 12h HTF regime filter to avoid counter-trend trades. Targets 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_regime_12h_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close for trend regime
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False).mean()
        wma_full = pd.Series(close_12h).ewm(span=21, adjust=False).mean()
        raw_hma = 2 * wma_half - wma_full
        hma_12h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # HMA(21) on 4h close for trend
    if n >= 21:
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close).ewm(span=half_len, adjust=False).mean()
        wma_full = pd.Series(close).ewm(span=21, adjust=False).mean()
        raw_hma = 2 * wma_half - wma_full
        hma_4h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
    else:
        hma_4h = np.full(n, np.nan)
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_4h[i]) or np.isnan(vol_ma[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in direction of 12h HMA trend ---
        # Long only when price > 12h HMA (bullish regime)
        # Short only when price < 12h HMA (bearish regime)
        bullish_regime = close[i] > hma_12h_aligned[i]
        bearish_regime = close[i] < hma_12h_aligned[i]
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_high[i-1]  # Break above upper channel
        breakout_down = close[i] < donchian_low[i-1]  # Break below lower channel
        
        # --- HMA Trend Alignment (4h) ---
        price_above_hma = close[i] > hma_4h[i]
        price_below_hma = close[i] < hma_4h[i]
        
        # --- Volume Confirmation ---
        volume_confirm = volume[i] > vol_ma[i]  # Above average volume
        
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
                # Take profit at 3R
                if close[i] >= entry_price + 3.0 * (entry_price - stop_level):
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
                # Take profit at 3R
                if close[i] <= entry_price - 3.0 * (stop_level - entry_price):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + price above 4h HMA + volume + bullish 12h regime
        if breakout_up and price_above_hma and volume_confirm and bullish_regime:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout down + price below 4h HMA + volume + bearish 12h regime
        elif breakout_down and price_below_hma and volume_confirm and bearish_regime:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals