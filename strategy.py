#!/usr/bin/env python3
"""
Experiment #654: 1h RSI(14) mean reversion with 4h/1d trend filter and volume confirmation
HYPOTHESIS: In ranging markets (common in 2025 BTC/ETH), RSI extremes reverse. 
Use 4h EMA(50) and 1d EMA(200) for trend alignment: only take mean-reversion trades 
in direction of higher timeframe trend. Volume spike confirms institutional interest.
Session filter (08-20 UTC) reduces noise. Target 15-37 trades/year with discrete sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_654_1h_rsi_meanrev_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute session hours ONCE (open_time is datetime64[ms]) ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for EMA(50) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for EMA(200) trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # sufficient for 1d EMA(200)
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Trend Alignment: 4h EMA50 and 1d EMA200 ---
        uptrend_4h = price > ema_4h_aligned[i]
        uptrend_1d = price > ema_1d_aligned[i]
        downtrend_4h = price < ema_4h_aligned[i]
        downtrend_1d = price < ema_1d_aligned[i]
        
        # --- RSI Mean Reversion Conditions ---
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~12h on 1h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: RSI oversold + uptrend alignment
            if rsi_oversold and uptrend_4h and uptrend_1d:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: RSI overbought + downtrend alignment
            elif rsi_overbought and downtrend_4h and downtrend_1d:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals