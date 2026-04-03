#!/usr/bin/env python3
"""
Experiment #003: 4h Donchian(20) breakout + 12h volume confirmation + 1d trend filter

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 12h volume spikes and aligned with 1d trend (price > EMA50 for longs, < EMA50 for shorts), capture high-momentum moves with institutional participation. The Donchian structure provides objective breakout levels, volume confirms conviction, and the 1d trend filter ensures we trade with the higher timeframe bias. Targets 20-50 trades/year on 4h (75-200 total over 4 years) to minimize fee drag while capturing strong trending moves. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_vol_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        # Warmup period: use expanding window
        for i in range(20):
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1d EMA50 ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using incremental true range
            if i >= 1:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                # Use Wilder's smoothing: ATR_t = (ATR_{t-1} * 13 + TR_t) / 14
                if i == warmup:
                    # Initialize ATR with first 14 periods
                    tr_sum = 0.0
                    for j in range(warmup-13, warmup+1):
                        if j >= 1:
                            tr_val = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                            tr_sum += tr_val
                    atr_14 = tr_sum / 14
                else:
                    atr_14 = (atr_14_prev * 13 + tr) / 14
                
                if position_side > 0:  # Long position
                    stop_level = entry_price - 2.5 * atr_14
                    if low[i] < stop_level:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        atr_14_prev = atr_14
                        continue
                    # Take profit at Donchian Low (trailing stop for longs)
                    if close[i] <= donchian_low[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        atr_14_prev = atr_14
                        continue
                else:  # Short position
                    stop_level = entry_price + 2.5 * atr_14
                    if high[i] > stop_level:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        atr_14_prev = atr_14
                        continue
                    # Take profit at Donchian High (trailing stop for shorts)
                    if close[i] >= donchian_high[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        atr_14_prev = atr_14
                        continue
                
                # Hold position
                signals[i] = position_side * SIZE
                atr_14_prev = atr_14
                continue
            else:
                signals[i] = position_side * SIZE
                continue
        
        # --- Initialize ATR for first warmup period ---
        if i == warmup:
            tr_sum = 0.0
            start_idx = max(1, warmup-13)
            for j in range(start_idx, warmup+1):
                tr_val = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr_sum += tr_val
            atr_14_prev = tr_sum / 14
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian High with volume confirmation and uptrend bias
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_1d_ema
        )
        
        # Short: Price breaks below Donchian Low with volume confirmation and downtrend bias
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_1d_ema
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