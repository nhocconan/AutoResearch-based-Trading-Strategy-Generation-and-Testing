#!/usr/bin/env python3
"""
Experiment #4154: 1h RSI(14) mean reversion with 4h EMA(50) trend filter + 1d volume spike
HYPOTHESIS: In 1h timeframe, mean reversion works best when aligned with higher timeframe trend. 
4h EMA(50) defines trend direction (bull/bear), 1d volume spike (>2.0x) confirms institutional interest, 
1h RSI(14) <30 for long, >70 for short provides mean reentry entries. Session filter (08-20 UTC) reduces noise. 
Target: 60-150 total trades over 4 years (15-37/year). Discrete sizing 0.20 minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4154_1h_rsi14_4h_ema50_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === Precompute session hours (08-20 UTC) ===
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h EMA(50) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d volume MA(20) for spike detection ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: RSI(14) for mean reversion ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 50.0), where=avg_loss!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, 20, 14) + 5  # EMA50, volMA20, RSI14 buffers
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * (high[i] - low[i]):  # Simple range-based stop
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * (high[i] - low[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike: current 1h volume > 2.0 * aligned 1d volume MA
        volume_spike = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if volume_spike:
            # Trend alignment: price vs 4h EMA(50)
            above_ema = price > ema_4h_aligned[i]
            below_ema = price < ema_4h_aligned[i]
            
            # Mean reversion: RSI extremes
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            
            # Long conditions: volume spike + above 4h EMA (uptrend) + RSI oversold
            long_entry = volume_spike and above_ema and rsi_oversold
            
            # Short conditions: volume spike + below 4h EMA (downtrend) + RSI overbought
            short_entry = volume_spike and below_ema and rsi_overbought
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
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