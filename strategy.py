#!/usr/bin/env python3
"""
Experiment #327: 6h ATR Channel Breakout + 1d Trend + Volume Confirmation

HYPOTHESIS: Price breaking above/below an ATR-based channel (mean ± 2*ATR) on 6h timeframe,
combined with 1d trend filter (price > EMA200 for long, < EMA200 for short) and volume confirmation
(volume > 1.5x 20-period average) captures high-probability breakouts in both bull and bear markets.
The ATR channel adapts to volatility, reducing false breakouts in ranging markets. Targets 12-37
trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_breakout_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(200) on 1d close
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate ATR(14) for channel
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate ATR-based channel: mean price ± 2*ATR
    # Use 20-period SMA of close as the mean
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    upper_channel = sma_20 + 2.0 * atr_14
    lower_channel = sma_20 - 2.0 * atr_14
    
    # Calculate volume ratio (current vs 20-period average) on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in direction of 1d EMA200 trend ---
        price_above_1d_ema = close[i] > ema_200_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_200_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss (use current ATR)
            atr_current = atr_14[i]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_current
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at upper channel (for long) or lower channel (for short)
                if close[i] >= upper_channel[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_current
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at lower channel (for short) or upper channel (for long)
                if close[i] <= lower_channel[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above upper channel with volume and trend alignment
        long_condition = (
            close[i] > upper_channel[i] and 
            price_above_1d_ema and 
            volume_spike
        )
        
        # Short: Price breaks below lower channel with volume and trend alignment
        short_condition = (
            close[i] < lower_channel[i] and 
            price_below_1d_ema and 
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