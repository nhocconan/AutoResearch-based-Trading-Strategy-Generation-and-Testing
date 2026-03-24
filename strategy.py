#!/usr/bin/env python3
"""
Experiment #055: 6h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Volume Confirm

Hypothesis: 6h timeframe sits between 4h and 12h - should capture multi-day trends
without excessive whipsaw. Previous 6h failures used overly complex regime filters
(CHOP, CRSI, weekly pivots) that blocked trades. This strategy uses SIMPLER logic:
- 6h HMA(21)/HMA(50) crossover for primary trend
- 12h HMA(50) for intermediate bias (not too restrictive)
- 1d price vs HMA(50) for major trend direction
- RSI(14) pullback entries (30-70 range, not extreme)
- Volume confirmation (above 20-bar average)
- Loose filters to ensure >=30 trades on train, >=3 on test

Why this might work where others failed:
- Simpler than CRSI/CHOP regime switches (those failed on 6h)
- HMA crossover is responsive but not as noisy as EMA
- RSI pullback (not extreme) ensures entries during trends
- Volume filter avoids low-liquidity false breakouts
- 12h/1d HTF provides bias without blocking all trades

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, >=3 test
Position size: 0.28 (28% of capital, conservative for 6h swings)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_vol_12h1d_v1"
timeframe = "6h"
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
    """Average True Range for stoploss"""
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
    """Simple moving average of volume"""
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for intermediate trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h and 1d HMA) ===
        # Loose bias: only block if strongly against
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_above_avg = volume[i] > vol_ma[i] * 0.8  # loose: 80% of avg is ok
        
        # === 6h HMA TREND ===
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # === HMA CROSSOVER SIGNALS ===
        # Detect crossover: hma_21 crosses above/below hma_50
        hma_cross_bull = (hma_21[i] > hma_50[i]) and (hma_21[i-1] <= hma_50[i-1])
        hma_cross_bear = (hma_21[i] < hma_50[i]) and (hma_21[i-1] >= hma_50[i-1])
        
        # === RSI PULLBACK ENTRY (not extreme, ensures trades) ===
        # Long: RSI pulled back to 35-50 in bull trend
        # Short: RSI pulled back to 50-65 in bear trend
        rsi_pullback_long = 30.0 < rsi[i] < 55.0
        rsi_pullback_short = 45.0 < rsi[i] < 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY CONDITIONS (loose to ensure trades)
        long_conditions = [
            hma_bull,  # Primary trend bull
            vol_above_avg,  # Volume confirmation
        ]
        # HTF bias: at least one HTF should be bull or neutral
        htf_long_ok = htf_12h_bull or htf_1d_bull or (not htf_12h_bear and not htf_1d_bear)
        
        # Entry trigger: HMA crossover OR RSI pullback in trend
        long_entry_trigger = hma_cross_bull or (hma_bull and rsi_pullback_long)
        
        if all(long_conditions) and htf_long_ok and long_entry_trigger:
            desired_signal = SIZE
        
        # SHORT ENTRY CONDITIONS (loose to ensure trades)
        short_conditions = [
            hma_bear,  # Primary trend bear
            vol_above_avg,  # Volume confirmation
        ]
        # HTF bias: at least one HTF should be bear or neutral
        htf_short_ok = htf_12h_bear or htf_1d_bear or (not htf_12h_bull and not htf_1d_bull)
        
        # Entry trigger: HMA crossover OR RSI pullback in trend
        short_entry_trigger = hma_cross_bear or (hma_bear and rsi_pullback_short)
        
        if all(short_conditions) and htf_short_ok and short_entry_trigger:
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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