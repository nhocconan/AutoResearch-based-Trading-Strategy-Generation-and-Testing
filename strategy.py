#!/usr/bin/env python3
"""
Experiment #074: 4h Primary + 12h HTF — HMA Trend + RSI Pullback + Vol Filter

Hypothesis: Simpler is better. Complex regime-switching (Chop + CRSI + Donchian) 
causes 0 trades (seen in #064, #065, #067). This uses proven components:
1. 12h HMA(21) for trend bias (HTF direction)
2. 4h RSI(14) pullback entries (buy dips in uptrend, sell rallies in downtrend)
3. 4h ATR(14) volatility filter (avoid dead markets)
4. 4h SMA(200) for major trend confirmation

Key differences from #069:
- Simpler HTF (12h HMA instead of 1d KAMA) - faster signal response
- Standard RSI(14) instead of CRSI - fewer parameters, more reliable
- Volatility filter ensures we only trade when there's movement
- Looser RSI thresholds (40-45 / 55-60) to ensure 30+ trades/symbol

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 20-50 trades/year = 80-200 over 4 year train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    wma_diff = 2.0 * wma1 - wma2
    
    hma = pd.Series(wma_diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility filter and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average for major trend filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volatility_ratio(atr, period=50):
    """ATR ratio to detect vol spikes vs normal"""
    n = len(atr)
    if n < period:
        return np.full(n, np.nan)
    
    atr_sma = pd.Series(atr).rolling(window=period, min_periods=period).mean().values
    vol_ratio = atr / atr_sma
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for HTF trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_ratio = calculate_volatility_ratio(atr_4h, period=50)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete, max 0.40 per rules)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(rsi_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_200[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND (HMA slope + price position) ===
        hma_slope_bull = hma_4h[i] > hma_4h[i-10] if i >= 10 else False
        hma_slope_bear = hma_4h[i] < hma_4h[i-10] if i >= 10 else False
        
        # === SMA200 TREND FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 40-45 in uptrend
        rsi_pullback_long = 38.0 <= rsi_4h[i] <= 48.0
        # Short: RSI rallied to 55-60 in downtrend
        rsi_pullback_short = 52.0 <= rsi_4h[i] <= 62.0
        
        # === VOLATILITY FILTER ===
        # Only trade when vol is at least 80% of 50-bar average (avoid dead markets)
        vol_ok = vol_ratio[i] >= 0.80
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: HTF bull + 4h trend + SMA200 + RSI pullback + vol ok
        if htf_bull and hma_slope_bull and above_sma200 and rsi_pullback_long and vol_ok:
            desired_signal = SIZE
        
        # SHORT: HTF bear + 4h trend + SMA200 + RSI pullback + vol ok
        elif htf_bear and hma_slope_bear and below_sma200 and rsi_pullback_short and vol_ok:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals