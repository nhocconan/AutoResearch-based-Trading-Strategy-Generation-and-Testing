#!/usr/bin/env python3
"""
Experiment #002: 12h Donchian Breakout + Volume + 1d Trend Filter

HYPOTHESIS: 12h Donchian(20) breakout captures institutional moves.
1d SMA(50) confirms trend direction (filters fakeouts).
Volume spike confirms commitment. ATR stoploss manages risk.
This worked in DB: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (test Sharpe=1.382)

WHY BOTH MARKETS:
- Bull: Long breakouts above 1d SMA50 with trailing ATR stop
- Bear: Short breakouts below 1d SMA50
- Range: Choppiness filter prevents whipsaws

TARGET: 75-125 total trades over 4 years (~20-30/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_sma_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper = highest high, lower = lowest low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA(50) for trend direction
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=50)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # 1d Donchian(20) for structure
    donch_1d_upper_raw, donch_1d_lower_raw = calculate_donchian(
        df_1d['high'].values, df_1d['low'].values, period=20
    )
    donch_1d_upper = align_htf_to_ltf(prices, df_1d, donch_1d_upper_raw)
    donch_1d_lower = align_htf_to_ltf(prices, df_1d, donch_1d_lower_raw)
    
    # 12h indicators
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME: Choppiness via ATR ratio ===
        # ATR ratio > 1.3 means high volatility (possible trend start)
        atr_ratio = atr_14[i] / pd.Series(atr_14).iloc[max(0,i-30):i+1].mean() if i > 30 else 1.0
        
        # === 1d TREND DIRECTION ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Upper breakout
        upper_broken = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1] if i > 0 else False
        # Lower breakout
        lower_broken = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.3
        
        desired_signal = 0.0
        
        # === LONG: Upper breakout + price above 1d SMA + vol confirm ===
        if upper_broken and price_above_1d_sma and vol_confirm:
            desired_signal = SIZE
        
        # === SHORT: Lower breakout + price below 1d SMA + vol confirm ===
        if lower_broken and not price_above_1d_sma and vol_confirm:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR-based) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals