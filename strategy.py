#!/usr/bin/env python3
"""
Experiment #002: 12h Donchian(20) Breakout + 1d Volume Spike + 1w Trend Filter

HYPOTHESIS: Donchian channel breakouts on 12h timeframe capture significant price moves,
confirmed by 1d volume spikes (>2x average) and aligned with 1w trend (price above/below EMA50).
This strategy targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) by using
tight entry conditions: breakout of 20-period Donchian channels with volume confirmation and
regime filter. Designed to work in both bull and bear markets by trading breakouts in the
direction of the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channels (20-period) ===
    # Calculate highest high and lowest low of past 20 periods on 12h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:  # Need 20 periods for calculation (0-indexed)
            start_idx = max(0, i - 19)
            highest_20[i] = np.max(high[start_idx:i+1])
            lowest_20[i] = np.min(low[start_idx:i+1])
        else:
            highest_20[i] = np.nan
            lowest_20[i] = np.nan
    
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
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in direction of 1w trend ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
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
                # Exit if price returns to middle of Donchian channel
                donchian_middle = (highest_20[i] + lowest_20[i]) / 2
                if abs(close[i] - donchian_middle) < (highest_20[i] - lowest_20[i]) * 0.1:
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
                # Exit if price returns to middle of Donchian channel
                donchian_middle = (highest_20[i] + lowest_20[i]) / 2
                if abs(close[i] - donchian_middle) < (highest_20[i] - lowest_20[i]) * 0.1:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume in uptrend
        long_condition = (
            close[i] > highest_20[i] and  # Breakout above upper band
            volume_spike and               # Volume confirmation
            price_above_1w_ema             # Aligned with 1w uptrend
        )
        
        # Short: Price breaks below Donchian lower band with volume in downtrend
        short_condition = (
            close[i] < lowest_20[i] and   # Breakdown below lower band
            volume_spike and               # Volume confirmation
            price_below_1w_ema             # Aligned with 1w downtrend
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
</trading_assistant>