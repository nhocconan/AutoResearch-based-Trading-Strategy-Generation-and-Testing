#!/usr/bin/env python3
"""
Experiment #214: 1h HTF(4h/1d) Trend + Volume + Session Filter

HYPOTHESIS: Use 4h EMA(21) trend direction and 1d higher high/low structure for bias, 
enter on 1h pullbacks with volume confirmation during active UTC 08-20 session. 
This captures swing trades in trends while avoiding chop and low-liquidity periods. 
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag. 
Discrete sizing (0.20) reduces turnover costs. Works in bull (trend continuation) 
and bear (trend reversals on higher timeframe structure).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_214_1h_htf_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h EMA(21) for trend direction (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 21:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d Higher High/Low Structure for bias ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Higher High: today's high > yesterday's high
        hh = df_1d['high'].values > np.roll(df_1d['high'].values, 1)
        # Higher Low: today's low > yesterday's low
        hl = df_1d['low'].values > np.roll(df_1d['low'].values, 1)
        # Lower High: today's high < yesterday's high
        lh = df_1d['high'].values < np.roll(df_1d['high'].values, 1)
        # Lower Low: today's low < yesterday's low
        ll = df_1d['low'].values < np.roll(df_1d['low'].values, 1)
        
        # Bullish structure: HH and HL
        bull_struct = hh & hl
        # Bearish structure: LH and LL
        bear_struct = lh & ll
        
        # Align to 1h timeframe with shift(1) for completed days only
        bull_struct_aligned = align_htf_to_ltf(prices, df_1d, bull_struct.astype(float))
        bear_struct_aligned = align_htf_to_ltf(prices, df_1d, bear_struct.astype(float))
    else:
        bull_struct_aligned = np.full(n, np.nan)
        bear_struct_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: EMA(21) for dynamic support/resistance ===
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade during active UTC 08-20 ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_21[i]) or 
            np.isnan(bull_struct_aligned[i]) or np.isnan(bear_struct_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- HTF Trend Bias: 4h EMA direction + 1d structure ---
        # Bullish bias: price above 4h EMA AND bullish 1d structure
        bullish_bias = (close[i] > ema_4h_aligned[i]) and (bull_struct_aligned[i] > 0.5)
        # Bearish bias: price below 4h EMA AND bearish 1d structure
        bearish_bias = (close[i] < ema_4h_aligned[i]) and (bear_struct_aligned[i] > 0.5)
        
        # --- 1h Pullback Entry: Price near 21 EMA with volume ---
        # Pullback to EMA: price within 0.5% of EMA(21)
        near_ema = abs(close[i] - ema_21[i]) / ema_21[i] < 0.005
        # Volume confirmation: above average volume
        volume_ok = vol_ratio[i] > 1.5
        
        # --- Exit Logic: Trailing stop via signal reversal ---
        if in_position:
            # Long exit: bearish bias OR price breaks below EMA with volume
            if position_side > 0:
                if bearish_bias or (close[i] < ema_21[i] and volume_ok):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            # Short exit: bullish bias OR price breaks above EMA with volume
            else:
                if bullish_bias or (close[i] > ema_21[i] and volume_ok):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bullish bias + pullback to EMA + volume confirmation
        long_condition = bullish_bias and near_ema and volume_ok
        # Short: Bearish bias + pullback to EMA + volume confirmation
        short_condition = bearish_bias and near_ema and volume_ok
        
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