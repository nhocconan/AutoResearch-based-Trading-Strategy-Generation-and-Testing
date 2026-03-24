#!/usr/bin/env python3
"""
Experiment #021: 4h Primary + 1d/1w HTF — HMA Triple Alignment + Vol-Adjusted Sizing

Hypothesis: After 20 failed experiments, the winning pattern is SIMPLER trend following
with STRONGER HTF alignment. Key insights from failures:
1. Complex regime switching (chop/connors/fisher) consistently fails → use pure trend
2. Too many filters = 0 trades → simplify entry logic
3. 4h timeframe works best → stick with it
4. KAMA worked (#011 Sharpe=0.221) but HMA may be faster for crypto volatility

New approach:
1. TRIPLE HMA alignment: 1w (major), 1d (intermediate), 4h (entry)
2. All 3 HMAs must agree for entry (stronger trend confirmation)
3. RSI(14) for entry timing only (not as hard filter) - enter on pullback
4. Volatility-adjusted position sizing: reduce size when ATR spikes
5. Trailing stop: 2.5x ATR from entry, tighten to 1.5x after 1.5R profit

Why this should beat #011 (KAMA Sharpe=0.221):
- 1w HTF adds major trend filter (missed in #011)
- HMA responds faster than KAMA to crypto moves
- Vol-adjusted sizing reduces DD in high-vol periods (2022 crash)
- Simpler logic = more trades, less overfitting

Entry Logic:
- Long: 4h close > 4h HMA + 1d close > 1d HMA + 1w close > 1w HMA + RSI(14) > 45
- Short: 4h close < 4h HMA + 1d close < 1d HMA + 1w close < 1w HMA + RSI(14) < 55
- Size: 0.25 base, reduced to 0.15 when ATR(14)/ATR(50) > 1.5 (vol spike)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.25, trades>30/symbol train, >3/symbol test, DD>-35%
Timeframe: 4h (target 25-45 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_triple_voladj_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - faster response than EMA with less lag
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Helper: Weighted Moving Average
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        weights /= weights.sum()
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i - span + 1:i + 1] * weights)
        return result
    
    half = period // 2
    if half < 1:
        half = 1
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA calculation
    hma_raw = np.zeros(n)
    for i in range(period - 1, n):
        if np.isnan(wma_half[i]) or np.isnan(wma_full[i]):
            hma_raw[i] = np.nan
        else:
            diff = 2.0 * wma_half[i] - wma_full[i]
            # Apply WMA with sqrt(period) window to the difference
            sqrt_period = int(np.sqrt(period))
            if sqrt_period < 1:
                sqrt_period = 1
            if i >= sqrt_period - 1:
                start_idx = i - sqrt_period + 1
                weights = np.arange(1, sqrt_period + 1, dtype=float)
                weights /= weights.sum()
                hma_raw[i] = np.sum(diff * weights) if start_idx >= 0 else np.nan
            else:
                hma_raw[i] = np.nan
    
    return hma_raw

def calculate_rsi(close, period=14):
    """RSI - momentum indicator for entry timing"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for volatility and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_long = calculate_atr(high, low, close, period=50)  # For vol regime
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size
    REDUCED_SIZE = 0.15  # Size during vol spikes
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    profit_target_hit = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY REGIME (adjust position size) ===
        vol_spike = False
        if not np.isnan(atr_long[i]) and atr_long[i] > 1e-10:
            vol_ratio = atr[i] / atr_long[i]
            if vol_ratio > 1.5:  # ATR 50% above long-term average
                vol_spike = True
        
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # === TRIPLE HMA ALIGNMENT ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === RSI ENTRY TIMING (loose thresholds for trade generation) ===
        rsi_ok_long = rsi[i] > 45.0  # Not in deep oversold (avoid catching falling knife)
        rsi_ok_short = rsi[i] < 55.0  # Not in deep overbought
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: All 3 HMAs bullish + RSI confirmation
        if hma_4h_bull and hma_1d_bull and hma_1w_bull and rsi_ok_long:
            desired_signal = current_size
        
        # Short entry: All 3 HMAs bearish + RSI confirmation
        elif hma_4h_bear and hma_1d_bear and hma_1w_bear and rsi_ok_short:
            desired_signal = -current_size
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            # Check profit target for trailing stop tightening
            unrealized_pnl = (close[i] - entry_price) / entry_atr
            if unrealized_pnl >= 1.5:
                profit_target_hit = True
            
            # Trailing stop: 2.5x ATR initially, 1.5x after 1.5R profit
            stop_mult = 1.5 if profit_target_hit else 2.5
            stop_price = highest_since_entry - stop_mult * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            unrealized_pnl = (entry_price - close[i]) / entry_atr
            if unrealized_pnl >= 1.5:
                profit_target_hit = True
            
            stop_mult = 1.5 if profit_target_hit else 2.5
            stop_price = lowest_since_entry + stop_mult * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= current_size * 0.85:
            final_signal = current_size
        elif desired_signal <= -current_size * 0.85:
            final_signal = -current_size
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
                profit_target_hit = False
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                profit_target_hit = False
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
                profit_target_hit = False
        
        signals[i] = final_signal
    
    return signals