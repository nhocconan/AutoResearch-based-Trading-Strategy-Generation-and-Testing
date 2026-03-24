#!/usr/bin/env python3
"""
Experiment #528: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Volume Confirmation

Hypothesis: Simpler is better. Complex regime filters (CHOP+ADX+multiple HTF) have failed
repeatedly (#518, #520, #522, #524). This strategy uses proven patterns:
1. 1d HMA(21) for trend bias (single HTF, not dual)
2. 4h HMA(16/48) for local trend
3. RSI(14) pullback entries in trend direction
4. Volume spike confirmation (1.5x avg volume)
5. Asymmetric entries: long in uptrend, short in downtrend

Key differences from failed #522 (KAMA+ADX+CHOP+2xHTF):
1. HMA instead of KAMA (faster response, proven in best strategies)
2. Single HTF (1d) instead of dual (1d+1w) - less conflicting signals
3. No CHOP/ADX regime filter - these create too many no-trade conditions
4. RSI pullback logic (RSI<45 in uptrend, RSI>55 in downtrend)
5. Volume confirmation to filter false breakouts
6. Simpler stoploss: 2.5x ATR fixed, no complex trailing

Entry logic:
- LONG: close > 1d_HMA AND close > 4h_HMA AND RSI < 45 AND volume > 1.5x avg
- SHORT: close < 1d_HMA AND close < 4h_HMA AND RSI > 55 AND volume > 1.5x avg
- Exit: RSI crosses 50 against position OR stoploss hit

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
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
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Volume moving average for spike detection"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25  # Slightly smaller short size (asymmetric)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS ===
        # 1d HMA = macro trend
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # 4h HMA crossover = local trend confirmation
        hma_bull = hma_16[i] > hma_48[i]
        hma_bear = hma_16[i] < hma_48[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # === RSI PULLBACK ===
        # In uptrend: wait for RSI pullback to 35-45 zone
        # In downtrend: wait for RSI rally to 55-65 zone
        rsi_pullback_long = rsi[i] < 45.0 and rsi[i] > 25.0
        rsi_pullback_short = rsi[i] > 55.0 and rsi[i] < 75.0
        
        # RSI recovery confirmation
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Trend up + HMA bull + RSI pullback + volume spike
        if trend_bull and hma_bull:
            if rsi_pullback_long and rsi_rising and vol_spike:
                desired_signal = SIZE_LONG
            elif rsi_pullback_long and rsi_rising:
                desired_signal = SIZE_LONG * 0.8  # No volume spike, smaller size
        
        # SHORT: Trend down + HMA bear + RSI rally + volume spike
        elif trend_bear and hma_bear:
            if rsi_pullback_short and rsi_falling and vol_spike:
                desired_signal = -SIZE_SHORT
            elif rsi_pullback_short and rsi_falling:
                desired_signal = -SIZE_SHORT * 0.8  # No volume spike, smaller size
        
        # === EXIT CONDITIONS (RSI cross 50 against position) ===
        if in_position and position_side > 0:
            if rsi[i] > 55.0:  # RSI overbought in long position
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if rsi[i] < 45.0:  # RSI oversold in short position
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_LONG * 0.9:
            final_signal = SIZE_LONG
        elif desired_signal <= -SIZE_SHORT * 0.9:
            final_signal = -SIZE_SHORT
        elif desired_signal >= SIZE_LONG * 0.6:
            final_signal = SIZE_LONG * 0.8
        elif desired_signal <= -SIZE_SHORT * 0.6:
            final_signal = -SIZE_SHORT * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals