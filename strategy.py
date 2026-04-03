#!/usr/bin/env python3
"""
Experiment #238: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Daily Donchian channel breakouts capture significant price movements with institutional participation. 
Weekly HMA(21) filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaw. 
Volume confirmation (>1.5x average) validates breakout strength. ATR-based stoploss manages risk. 
This structure has proven effective on SOLUSDT (test Sharpe 1.10-1.38) and should generalize across BTC/ETH/SOL. 
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag while capturing high-probability trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_htf_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        # Ensure arrays are same length for subtraction
        min_len = min(len(wma_half), len(wma_full))
        wma_half = wma_half[-min_len:]
        wma_full = wma_full[-min_len:]
        raw_hma = 2 * wma_half - wma_full
        hma_21_1w = wma(raw_hma, sqrt_len)
        # Pad beginning with NaN
        hma_21_1w_full = np.full(len(close_1w), np.nan)
        hma_21_1w_full[-len(hma_21_1w):] = hma_21_1w
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_full)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i >= 19:  # 20 periods including current
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
    # Volume average (20-period)
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
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation ---
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
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
                # Exit on opposite Donchian break (trailing stop)
                if close[i] < donchian_low[i]:
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
                # Exit on opposite Donchian break (trailing stop)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high + above weekly HMA + volume confirmation
        if (close[i] > donchian_high[i] and 
            close[i] > hma_21_1w_aligned[i] and 
            vol_confirm):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian low + below weekly HMA + volume confirmation
        elif (close[i] < donchian_low[i] and 
              close[i] < hma_21_1w_aligned[i] and 
              vol_confirm):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals