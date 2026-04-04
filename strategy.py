#!/usr/bin/env python3
"""
Experiment #5252: 12h Donchian Breakout + Volume Spike + Regime Filter
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts from 1d timeframe provide institutional-grade support/resistance. 
Enter long when price breaks above 1d Donchian(20) upper band with volume spike (>2.0x) in bullish regime (price > 1w EMA50). 
Enter short when price breaks below 1d Donchian(20) lower band with volume spike in bearish regime (price < 1w EMA50). 
Breakouts are filtered by regime to avoid counter-trend entries. Uses ATR-based trailing stop (2.0*ATR) to manage risk. 
Designed for 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag. Works in both bull and bear markets 
by only trading in direction of higher timeframe trend (1w EMA50). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5252_12h_donchian_breakout_vol_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian channels and regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Donchian Channel (20) ===
    if len(df_1d) >= 20:
        # Upper band: highest high of last 20 days (prior 20, not including current)
        donchian_upper = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
        # Lower band: lowest low of last 20 days (prior 20, not including current)
        donchian_lower = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
        
        # Align to 12h timeframe (shift(1) in align_htf_to_ltf ensures prior completed day only)
        donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
        donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    else:
        donchian_upper_aligned = np.full(n, np.nan)
        donchian_lower_aligned = np.full(n, np.nan)
    
    # === 1w Indicators: EMA50 for regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        ema_50 = pd.Series(df_1w['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Volume confirmation (2.0x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian warmup, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Regime filter: bullish if price > 1w EMA50, bearish if price < 1w EMA50
        regime_bullish = price > ema_50_aligned[i]
        regime_bearish = price < ema_50_aligned[i]
        
        # Donchian breakout: long on upper band break, short on lower band break
        breakout_long = (price >= donchian_upper_aligned[i]) and regime_bullish and vol_confirm
        breakout_short = (price <= donchian_lower_aligned[i]) and regime_bearish and vol_confirm
        
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

</think>