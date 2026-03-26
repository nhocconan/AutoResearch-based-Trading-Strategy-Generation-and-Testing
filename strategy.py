#!/usr/bin/env python3
"""
Experiment #024: 6h RSI(2) Mean Reversion + ATR Volatility Expansion + 1d Trend

HYPOTHESIS: RSI(2) captures short-term overextensions that revert. Combined with
ATR expansion confirmation (>1.5x 20-bar ATR) to avoid chop, and 1d EMA(50) for 
trend alignment, this catches reversals at major support/resistance. Works in 
both bull (long RSI<20 bounces) and bear (short RSI>80 rejections) markets.

TIMEFRAME: 6h primary
HTF: 1d for trend alignment
TARGET: 75-150 total trades over 4 years (12-37/year)
SIZE: 0.25 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_rsi2_atr_expansion_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """RSI indicator"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period):
    """EMA indicator"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend
    ema_1d_raw = calculate_ema(df_1d['close'].values, period=50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # Local indicators
    rsi_2 = calculate_rsi(close, period=2)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_20 = calculate_atr(high, low, close, period=20)
    
    # ATR expansion ratio (current ATR vs 20-bar ATR)
    atr_ratio = atr_14 / np.where(atr_20 > 0, atr_20, 1)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        rsi2 = rsi_2[i]
        rsi14 = rsi_14[i]
        atr_expansion = atr_ratio[i]
        vol_spike = vol_ratio[i] > 1.2
        
        # === TREND DIRECTION (1d EMA) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI(2) oversold + ATR expansion + trend aligned ===
            # RSI(2) < 15 indicates extreme oversold
            # ATR expansion > 1.5 confirms volatility spike (potential reversal)
            # Price above 1d EMA confirms bullish trend
            if rsi2 < 15 and atr_expansion > 1.5 and price_above_1d_ema:
                desired_signal = SIZE
            
            # === SHORT: RSI(2) overbought + ATR expansion + trend opposed ===
            # RSI(2) > 85 indicates extreme overbought
            # Price below 1d EMA confirms bearish trend
            if rsi2 > 85 and atr_expansion > 1.5 and not price_above_1d_ema:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: RSI(2) mean reverts OR opposite extreme ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: RSI(2) mean reverts above 50
            if rsi2 > 50:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: RSI(2) mean reverts below 50
            if rsi2 < 50:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals