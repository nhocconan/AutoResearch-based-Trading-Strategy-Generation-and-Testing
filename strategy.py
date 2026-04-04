#!/usr/bin/env python3
"""
Experiment #4214: 1h RSI(14) mean reversion with 4h trend filter and 1d volume spike
HYPOTHESIS: On 1h timeframe, buy when RSI < 30 (oversold) in uptrend (price > 4h EMA50) and sell when RSI > 70 (overbought) in downtrend (price < 4h EMA50), confirmed by 1d volume > 1.5x average. Use 4h/1d for signal direction, 1h only for entry timing. Session filter 08-20 UTC to avoid low-liquidity hours. Discrete position size 0.20 targets 60-150 trades over 4 years (15-37/year). ATR-based stoploss (2.0x) manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4214_1h_rsi14_4h_ema50_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values  # already datetime64[ms]
    n = len(close)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 4h EMA50 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === Precompute HTF: 1d volume MA(20) for confirmation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: RSI(14) ===
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[np.nan], rsi])
    
    rsi = calculate_rsi(close, 14)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(14, 14, 50, 20)  # RSI, ATR, 4h EMA50, 1d vol MA
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(ema_4h_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
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
        # Volume confirmation: 1d volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if volume_confirm:
            # RSI conditions
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            
            # 4h EMA50 trend filter
            price_above_ema = price > ema_4h_aligned[i]
            price_below_ema = price < ema_4h_aligned[i]
            
            # Long conditions: RSI oversold + price above 4h EMA50 (uptrend)
            long_entry = rsi_oversold and price_above_ema
            
            # Short conditions: RSI overbought + price below 4h EMA50 (downtrend)
            short_entry = rsi_overbought and price_below_ema
            
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