#!/usr/bin/env python3
"""
Experiment #3754: 1h EMA(21) pullback to 4h trend with 1d EMA(50) filter + volume confirmation
HYPOTHESIS: In 1h timeframe, trade pullbacks to the 21 EMA in the direction of the 4h trend (EMA50 > EMA200) and 1d regime (price > 1d EMA50 for longs, < for shorts). Volume confirmation (>1.5x average) ensures momentum. Uses ATR(14) trailing stop (2.0x) to manage drawdown. Position size 0.20 balances risk and return. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3754_1h_ema21_pullback_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === 1h Indicators: EMA(21) for pullback entries ===
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === HTF: 4h data for trend direction (EMA50 > EMA200) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    trend_4h_up = ema_50_4h > ema_200_4h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up)
    
    # === HTF: 1d data for regime filter (price vs EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    price_above_1d_ema50 = close_1d > ema_50_1d
    price_above_1d_ema50_aligned = align_htf_to_ltf(prices, df_1d, price_above_1d_ema50)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(21, 14, 20, 50, 200, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_21[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_4h_up_aligned[i]) or np.isnan(price_above_1d_ema50_aligned[i])):
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
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: Price near 21 EMA (pullback) in uptrend regime
            # Price within 0.5% of EMA21 (pullback zone)
            near_ema21 = abs(price - ema_21[i]) / ema_21[i] < 0.005
            # 4h trend up AND price above 1d EMA50 (bullish regime)
            if near_ema21 and trend_4h_up_aligned[i] and price_above_1d_ema50_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price near 21 EMA (pullback) in downtrend regime
            elif near_ema21 and not trend_4h_up_aligned[i] and not price_above_1d_ema50_aligned[i]:
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