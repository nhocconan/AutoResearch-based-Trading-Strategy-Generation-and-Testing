#!/usr/bin/env python3
"""
Experiment #5065: 12h Donchian(20) Breakout + 1d ATR/Volume Regime Filter + ATR Stoploss
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts filtered by 1d ATR expansion (volatility regime) and volume spikes capture strong momentum moves while avoiding choppy markets. The 1d ATR ratio (current ATR / 20-period MA ATR) acts as a regime filter: >1.2 indicates expanding volatility favorable for breakouts. Volume > 1.5x 20-period average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 12-37 trades/year on 12h timeframe to minimize fee drag while maintaining statistical significance. Works in both bull (breakouts with expansion) and bear (breakdowns with expansion) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5065_12h_donchian20_1d_atr_vol_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ATR Regime Filter ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values.astype(np.float64)
        low_1d = df_1d['low'].values.astype(np.float64)
        close_1d = df_1d['close'].values.astype(np.float64)
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # 20-period MA of ATR
        atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
        
        # ATR Ratio: current ATR / MA(ATR) > 1.2 indicates expanding volatility regime
        atr_ratio_1d = np.ones_like(atr_1d)
        atr_ratio_1d[20:] = atr_1d[20:] / atr_ma_1d[20:]
        
        # Align to 12h timeframe
        atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    else:
        atr_ratio_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation (1.5x spike) ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Regime filter: 1d ATR expansion (>1.2)
        regime_filter = atr_ratio_aligned[i] > 1.2
        
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = (price >= high_roll[i]) and regime_filter and vol_confirm
        breakout_short = (price <= low_roll[i]) and regime_filter and vol_confirm
        
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