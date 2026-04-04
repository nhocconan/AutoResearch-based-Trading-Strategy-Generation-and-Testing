#!/usr/bin/env python3
"""
Experiment #5253: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: On 4h timeframe, Donchian channel breakouts (20-period) provide high-probability trend entries when aligned with 12h HMA(21) trend filter and volume confirmation (>1.5x average). Designed for 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag. Works in bull markets (breakouts above upper band with volume) and bear markets (breakouts below lower band with volume). Uses discrete position sizing (0.30) to balance return and drawdown. ATR-based trailing stop (2.0) limits losses during reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5253_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: HMA(21) for trend ===
    if len(df_12h) >= 21:
        # Hull Moving Average calculation
        n_hl = int(21 / 2)
        n_sqrt = int(np.sqrt(21))
        
        # WMA helper function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_wma = wma(df_12h['close'].values, n_hl)
        full_wma = wma(df_12h['close'].values, 21)
        raw_hma = 2 * half_wma - full_wma
        hma_21 = wma(raw_hma, n_sqrt)
        
        # Align to 4h timeframe (shift(1) in align_htf_to_ltf ensures prior completed bar only)
        hma_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    else:
        hma_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 4h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(donchian_window, 20, 21, 14)  # Donchian, volume MA, HMA warmup, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(hma_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: bullish if price > 12h HMA(21), bearish if price < 12h HMA(21)
        trend_bullish = price > hma_aligned[i]
        trend_bearish = price < hma_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = (price >= upper[i]) and trend_bullish and vol_confirm
        breakout_short = (price <= lower[i]) and trend_bearish and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals