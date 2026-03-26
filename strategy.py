#!/usr/bin/env python3
"""
Experiment #021: 12h TRIX Momentum + Volume + 1d SMA200 Trend

HYPOTHESIS: TRIX (Triple EMA) measures rate of change of momentum. 
A TRIX crossover signals momentum shift — more robust than simple EMA crossover.
Combined with volume confirmation and 1d SMA200 trend filter, this captures 
trend changes without overtrading. TRIX is a smoothing oscillator that filters 
noise better than raw momentum indicators. 12h TF = ~20 bars/week = ~75-150 trades/4yr.

TIMEFRAME: 12h primary
HTF: 1d for trend direction filter
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trix_vol_1d_sma200_v1"
timeframe = "12h"
leverage = 1.0

def calculate_trix(close, period=14):
    """TRIX - Triple EMA Rate of Change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple EMA
    trix = ema3.pct_change(period) * 100
    
    return trix.values

def calculate_sma(close, period):
    """Simple Moving Average"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma200_1d_raw = calculate_sma(df_1d['close'].values, period=200)
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # TRIX
    trix = calculate_trix(close, period=14)
    
    # TRIX signal line (EMA of TRIX)
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma200_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === 1d TREND FILTER ===
        bullish_trend = close[i] > sma200_1d_aligned[i]
        bearish_trend = close[i] < sma200_1d_aligned[i]
        
        # === TRIX CROSSOVER DETECTION ===
        # Bullish crossover: TRIX crosses above signal line
        trix_above_sig_prev = trix[i-1] > trix_signal[i-1] if i > 1 else False
        trix_above_sig_curr = trix[i] > trix_signal[i]
        bullish_crossover = trix_above_sig_curr and not trix_above_sig_prev
        
        # Bearish crossover: TRIX crosses below signal line
        trix_below_sig_prev = trix[i-1] < trix_signal[i-1] if i > 1 else False
        trix_below_sig_curr = trix[i] < trix_signal[i]
        bearish_crossover = trix_below_sig_curr and not trix_below_sig_prev
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI MOMENTUM ===
        rsi_val = rsi[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # TRIX bullish crossover + volume spike + bullish 1d trend + RSI confirming
            if bullish_crossover and vol_spike and bullish_trend and rsi_val > 45:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # TRIX bearish crossover + volume spike + bearish 1d trend + RSI confirming
            if bearish_crossover and vol_spike and bearish_trend and rsi_val < 55:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === EXIT: Opposite crossover or RSI extreme ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: TRIX bearish crossover OR RSI < 35
            if bearish_crossover:
                exit_triggered = True
            if rsi_val < 35:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: TRIX bullish crossover OR RSI > 65
            if bullish_crossover:
                exit_triggered = True
            if rsi_val > 65:
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
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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