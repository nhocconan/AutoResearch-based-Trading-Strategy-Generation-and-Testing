#!/usr/bin/env python3
"""
Experiment #006: 4h RSI(14) Extreme + 1d SMA(50) Trend + ATR Stop

HYPOTHESIS: RSI extremes (< 35 or > 65) capture mean reversion opportunities
while 1d SMA(50) filter ensures entries align with the primary trend.
This works in BOTH bull and bear:
- Bull: RSI < 35 near SMA = oversold rally opportunity
- Bear: RSI > 65 below SMA = mean reversion shorts

WHY IT SHOULD WORK:
- RSI extremes are classic, well-understood signals
- 1d SMA gives clear trend direction without overfitting
- ATR stoploss handles volatility regime changes
- Simple 3 conditions = 100-200 trades expected

TARGET: 100-200 total trades over 4 years (25-50/year) — proven range
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_extreme_1d_sma_atr_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=14):
    """RSI(14)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA(50) for trend direction
    sma_1d = calculate_sma(df_1d['close'].values, period=50)
    sma_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)  # auto shift(1) for completed bars
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume ratio (20-period MA) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # 200 for SMA + indicator safety margin
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1d TREND FILTER ===
        price_above_sma = close[i] > sma_aligned[i]   # Bullish
        price_below_sma = close[i] < sma_aligned[i]   # Bearish
        
        # === RSI EXTREME CONDITIONS ===
        rsi_oversold = rsi_14[i] < 35  # Strong oversold
        rsi_overbought = rsi_14[i] > 65  # Strong overbought
        
        # === VOLUME CONFIRMATION (optional boost) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI oversold + price above SMA + optional volume ===
            if rsi_oversold and price_above_sma:
                desired_signal = SIZE
            
            # === SHORT: RSI overbought + price below SMA + optional volume ===
            if rsi_overbought and price_below_sma:
                desired_signal = -SIZE
        
        # === STOPLOSS (2x ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2x ATR from recent high
                stop_price = trailing_high - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if trend flips (price below SMA)
                if price_below_sma:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2x ATR from recent low
                stop_price = trailing_low + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if trend flips (price above SMA)
                if price_above_sma:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 6 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals