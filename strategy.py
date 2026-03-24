#!/usr/bin/env python3
"""
Experiment #204: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous 12h strategies failed due to OVERLY COMPLEX entry conditions
that generated ZERO trades. This version SIMPLIFIES drastically:

1. Remove Choppiness Index regime switching (too many no-trade zones)
2. Remove Connors RSI (extreme thresholds = no signals)
3. Use standard RSI(14) with generous thresholds (35/65 vs 15/85)
4. Single regime: trend-following with pullback entries
5. 1d HTF for bias only, not hard filter

Key insight from 182 failed experiments: ENTRY CONDITIONS MUST BE LOOSE ENOUGH
to generate 20-50 trades/year. Complex confluence = 0 trades = auto-reject.

Entry Logic:
- Long: 12h HMA bullish + RSI(14) pulls back to 35-50 + 1d trend neutral/bull
- Short: 12h HMA bearish + RSI(14) rallies to 50-65 + 1d trend neutral/bear
- Exit: RSI crosses 50 against position OR 2.5x ATR stoploss

Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
Stoploss: Signal → 0 when price moves 2.5x ATR against position

Target: Sharpe>0.40 (beat current best 0.399), trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d_v1"
timeframe = "12h"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h_fast = calculate_hma(close, period=16)
    hma_12h_slow = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        if np.isnan(hma_12h_fast[i]) or np.isnan(hma_12h_slow[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF BIAS (1d HMA50) ===
        # More lenient: just check slope/direction, not hard filter
        htf_bull = hma_1d_aligned[i] > hma_1d_aligned[i-1] if not np.isnan(hma_1d_aligned[i-1]) else True
        htf_bear = hma_1d_aligned[i] < hma_1d_aligned[i-1] if not np.isnan(hma_1d_aligned[i-1]) else True
        
        # === 12h HMA TREND (fast vs slow crossover) ===
        hma_bull = hma_12h_fast[i] > hma_12h_slow[i]
        hma_bear = hma_12h_fast[i] < hma_12h_slow[i]
        
        # === RSI PULLBACK ZONES ===
        # Long: RSI pulled back to 35-50 in uptrend
        # Short: RSI rallied to 50-65 in downtrend
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        rsi_extreme_long = rsi[i] < 40.0
        rsi_extreme_short = rsi[i] > 60.0
        
        # === SMA200 FILTER (soft) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - generate more trades) ===
        desired_signal = 0.0
        
        # LONG entries (multiple conditions, any can trigger)
        if hma_bull:
            # Base long: HMA bull + RSI pullback
            if rsi_pullback_long:
                if htf_bull and above_sma200:
                    desired_signal = SIZE_STRONG
                elif above_sma200:
                    desired_signal = SIZE_BASE
                elif htf_bull:
                    desired_signal = SIZE_BASE
            
            # Extreme long: RSI very oversold in uptrend
            if rsi_extreme_long and hma_bull:
                desired_signal = max(desired_signal, SIZE_BASE)
        
        # SHORT entries (multiple conditions, any can trigger)
        if hma_bear:
            # Base short: HMA bear + RSI rally
            if rsi_pullback_short:
                if htf_bear and below_sma200:
                    desired_signal = -SIZE_STRONG
                elif below_sma200:
                    desired_signal = -SIZE_BASE
                elif htf_bear:
                    desired_signal = -SIZE_BASE
            
            # Extreme short: RSI very overbought in downtrend
            if rsi_extreme_short and hma_bear:
                desired_signal = min(desired_signal, -SIZE_BASE)
        
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
        
        # === EXIT LOGIC (RSI cross 50 against position) ===
        if in_position and position_side > 0 and rsi[i] > 60.0:
            # Long position, RSI overbought = take profit
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi[i] < 40.0:
            # Short position, RSI oversold = take profit
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
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