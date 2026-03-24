#!/usr/bin/env python3
"""
Experiment #203: 6h Primary + 1d/1w HTF — Triple HMA Trend + RSI Pullback

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). 
Previous 6h attempts failed due to OVERLY STRICT entry conditions (0 trades).
This version SIMPLIFIES entry logic to ensure adequate trade frequency:

Core Logic:
- 1w HMA(50): Ultra-long-term bias (only trade with weekly trend)
- 1d HMA(21): Major trend direction filter
- 6h HMA(21): Primary trend + pullback entry zone
- RSI(7): Faster response than RSI(14), entries at 35/65 (not 20/80)

Entry Conditions (LOOSER than failed attempts):
- Long: 1w HMA bull + 1d HMA bull + 6h price > 6h HMA + RSI(7) crosses above 35
- Short: 1w HMA bear + 1d HMA bear + 6h price < 6h HMA + RSI(7) crosses below 65

Why this should work:
1. Triple HMA alignment ensures we trade with multi-timeframe trend
2. RSI(7) at 35/65 triggers MORE often than RSI(14) at 20/80
3. 6h timeframe = 4 candles/day = ~1460 candles/year = 30-60 trades achievable
4. Simpler logic = fewer conditions that can all fail simultaneously

Position sizing: 0.25 base, 0.30 for strong HTF alignment
Stoploss: 2.0x ATR trailing (tighter than 2.5x to reduce DD)

Target: Sharpe>0.40 (beat current 6h best), DD>-35%, trades>=120 train, trades>=15 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_triple_hma_rsi_pullback_1d1w_v1"
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

def calculate_momentum(close, period=10):
    """Rate of Change momentum"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    mom = np.zeros(n)
    mom[:] = np.nan
    for i in range(period, n):
        mom[i] = (close[i] - close[i-period]) / close[i-period] * 100.0
    
    return mom

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultra-long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for major trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for more signals
    momentum = calculate_momentum(close, period=10)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track RSI crosses for entry timing
    prev_rsi = np.nan
    
    for i in range(150, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        # === HTF BIAS (1w and 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === RSI CROSS DETECTION ===
        rsi_cross_above_35 = False
        rsi_cross_below_65 = False
        
        if not np.isnan(prev_rsi) and not np.isnan(rsi[i]):
            rsi_cross_above_35 = (prev_rsi < 35.0 and rsi[i] >= 35.0)
            rsi_cross_below_65 = (prev_rsi > 65.0 and rsi[i] <= 65.0)
        
        # === MOMENTUM CONFIRMATION ===
        mom_positive = False
        mom_negative = False
        if not np.isnan(momentum[i]):
            mom_positive = momentum[i] > 0.5  # Small positive momentum
            mom_negative = momentum[i] < -0.5  # Small negative momentum
        
        # === ENTRY LOGIC (LOOSER conditions for more trades) ===
        desired_signal = 0.0
        
        # LONG ENTRY: Weekly bull + Daily bull + 6h pullback complete + RSI cross
        if htf_1w_bull and htf_1d_bull:
            # Strong signal: All HTF align + RSI cross + momentum
            if rsi_cross_above_35 and mom_positive:
                desired_signal = SIZE_STRONG
            # Base signal: HTF align + RSI cross (momentum optional)
            elif rsi_cross_above_35 and hma_6h_bull:
                desired_signal = SIZE_BASE
            # Weaker signal: HTF align + price above 6h HMA + RSI rising
            elif hma_6h_bull and rsi[i] > 40.0 and not np.isnan(prev_rsi) and rsi[i] > prev_rsi:
                desired_signal = SIZE_BASE * 0.6
        
        # SHORT ENTRY: Weekly bear + Daily bear + 6h rally complete + RSI cross
        elif htf_1w_bear and htf_1d_bear:
            # Strong signal: All HTF align + RSI cross + momentum
            if rsi_cross_below_65 and mom_negative:
                desired_signal = -SIZE_STRONG
            # Base signal: HTF align + RSI cross (momentum optional)
            elif rsi_cross_below_65 and hma_6h_bear:
                desired_signal = -SIZE_BASE
            # Weaker signal: HTF align + price below 6h HMA + RSI falling
            elif hma_6h_bear and rsi[i] < 60.0 and not np.isnan(prev_rsi) and rsi[i] < prev_rsi:
                desired_signal = -SIZE_BASE * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS (RSI extreme opposite) ===
        if in_position and position_side > 0 and rsi[i] > 75.0:
            # Long position, RSI overbought - reduce or exit
            if desired_signal == 0.0:
                desired_signal = 0.0  # Exit on stoploss
        
        if in_position and position_side < 0 and rsi[i] < 25.0:
            # Short position, RSI oversold - reduce or exit
            if desired_signal == 0.0:
                desired_signal = 0.0  # Exit on stoploss
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.6
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.6
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
        prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
    
    return signals