#!/usr/bin/env python3
"""
Experiment #3594: 1h Donchian Breakout + 4h EMA Trend + 1d Volume Regime
HYPOTHESIS: 4h EMA establishes trend direction, 1h Donchian(20) breakout provides entry timing with volume confirmation from 1d volume regime. Session filter (08-20 UTC) reduces noise. Position size 0.20. Target: 60-150 total trades over 4 years (15-37/year). Uses 4h for trend filter, 1d for volume regime, 1h for entry/exit timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3594_1h_donchian20_4h_ema_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = prices.index.hour
    
    # === HTF: 4h data for EMA trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(21) for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for volume regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA(20) for regime
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime_1d = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1h Indicators: Donchian Channel (20) for breakout ===
    lookback_donchian = 20
    highest_high = pd.Series(high).rolling(window=lookback_donchian, min_periods=lookback_donchian).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_donchian, min_periods=lookback_donchian).min().values
    
    # === 1h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_donchian + 1, 21, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_regime_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio_1d = volume[i] / vol_regime_1d[i] if vol_regime_1d[i] > 0 else 1.0
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (20) - trend reversal
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (20) - trend reversal
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume > 1.5x 1d average for confirmation
        volume_confirm = vol_ratio_1d > 1.5
        
        if volume_confirm:
            # Determine trend bias from 4h EMA
            bullish_bias = ema_4h_aligned[i] > close_4h[-1] if len(close_4h) > 0 else ema_4h_aligned[i] > price
            
            # Long entry: price breaks above Donchian high (20) in bullish 4h trend
            if (price > highest_high[i] and 
                bullish_bias):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low (20) in bearish 4h trend
            elif (price < lowest_low[i] and 
                  not bullish_bias):
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals