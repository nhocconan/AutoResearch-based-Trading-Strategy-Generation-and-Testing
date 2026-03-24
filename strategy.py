#!/usr/bin/env python3
"""
Experiment #1610: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + Pullback Entry

Hypothesis: After 11 failed experiments with CRSI/Choppiness on lower TFs, the issue is 
OVER-FILTERING. Strategies with too many confluence filters generate 0 trades (Sharpe=0.000).

This strategy uses:
1. EHLERS FISHER TRANSFORM - better reversal detection than RSI in bear/range markets
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. 4h HMA(21) - trend direction bias (simple, proven edge)
3. 12h HMA(21) - intermediate trend confirmation
4. 1h RSI(14) pullback - entry timing within HTF trend (loose: 35-65 zone)
5. ATR(14) 2.5x trailing stop - drawdown control
6. LOOSE entry thresholds - ensure minimum 30 trades/year per symbol

Key insight from failures: CRSI <10/>90 is TOO EXTREME for 1h. RSI 35-65 pullback 
in direction of 4h trend generates consistent trades with positive expectancy.

Timeframe: 1h (required for this experiment)
HTF: 4h HMA + 12h HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 30/symbol train, > 3/symbol test, DD > -50%
Trade Frequency: Target 40-80 trades/year (1h with HTF filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_pullback_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points better than RSI in bear/range markets
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Calculate median price
    median = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        x = (median[i] - lowest) / range_val
        x = max(0.001, min(0.999, x))  # clamp to avoid log(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Smooth with EMA
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        fisher_prev[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, fisher_prev

def calculate_rsi(close, period=14):
    """Standard RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period=21):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for primary trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for intermediate trend confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (4h + 12h HMA) ===
        # Long bias: price above both 4h and 12h HMA
        trend_long = close[i] > hma_4h_aligned[i] and close[i] > hma_12h_aligned[i]
        # Short bias: price below both 4h and 12h HMA
        trend_short = close[i] < hma_4h_aligned[i] and close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNAL ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_prev[i] <= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_prev[i] >= 1.5
        
        # === RSI PULLBACK ENTRY (loose thresholds for trade generation) ===
        # Long: RSI pulled back to 35-50 zone in uptrend
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        # Short: RSI rallied to 50-65 zone in downtrend
        rsi_pullback_short = 50.0 <= rsi[i] <= 65.0
        
        # === EMA MOMENTUM CONFIRMATION ===
        # Long: price above EMA21 (momentum aligned)
        ema_long = close[i] > ema_21[i]
        # Short: price below EMA21 (momentum aligned)
        ema_short = close[i] < ema_21[i]
        
        # === PRIMARY SIGNAL LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Trend up + Fisher reversal OR RSI pullback + EMA confirmation
        if trend_long:
            # Entry trigger: Fisher reversal OR (RSI pullback + EMA aligned)
            if fisher_long or (rsi_pullback_long and ema_long):
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Trend down + Fisher reversal OR RSI pullback + EMA confirmation
        elif trend_short:
            # Entry trigger: Fisher reversal OR (RSI pullback + EMA aligned)
            if fisher_short or (rsi_pullback_short and ema_short):
                desired_signal = -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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
                # Position flip
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