#!/usr/bin/env python3
"""
Experiment #257: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts capture strong momentum moves, while 4h HMA(21) filters for trend alignment and volume confirmation ensures institutional participation. ATR-based stoploss manages risk. This combination has proven effective on SOLUSDT (test Sharpe 1.10-1.38) and should work across BTC/ETH/SOL in both bull and bear markets by targeting strong trending moves. Targets 25-50 trades/year on 4h timeframe (100-200 total over 4 years) to minimize fee drag while capturing high-probability breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_1d_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        raw_hma = 2 * wma_half - wma_full
        hma_1d = wma(raw_hma, sqrt_len)
        
        # Align to 4h timeframe
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio: current volume / average volume(20)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(avg_volume > 0, volume / avg_volume, 0)
    
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
            np.isnan(hma_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss or mean reversion) ---
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
                # Exit on Donchian mean reversion (touch opposite band)
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
                # Exit on Donchian mean reversion (touch opposite band)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: above average volume
        vol_confirmed = volume_ratio[i] > 1.5
        
        # Long: Donchian breakout above upper band + price above 1d HMA (uptrend) + volume
        if (close[i] > donchian_high[i] and 
            close[i] > hma_1d_aligned[i] and 
            vol_confirmed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout below lower band + price below 1d HMA (downtrend) + volume
        elif (close[i] < donchian_low[i] and 
              close[i] < hma_1d_aligned[i] and 
              vol_confirmed):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals