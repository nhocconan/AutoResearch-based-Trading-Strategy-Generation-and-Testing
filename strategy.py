#!/usr/bin/env python3
"""
Experiment #014: 4h Primary + 12h HTF — Dual Regime (Choppiness Switch)

Hypothesis: The KAMA+ADX approach (#011) works but ADX>20 filter kills trades 
in range markets. Research shows Choppiness Index can switch between trend-follow 
and mean-revert modes. This dual-regime approach should:
1. Generate MORE trades (ADX was too restrictive)
2. Adapt to market conditions (trend vs range)
3. Work across BTC/ETH/SOL (not just SOL-biased)

Regime Logic:
- CHOP(14) < 45: TREND regime → follow 4h HMA direction + 12h HMA bias
- CHOP(14) > 55: RANGE regime → RSI mean-reversion (RSI<35 long, RSI>65 short)
- CHOP 45-55: Transition zone → stay flat or reduce size

Entry Conditions (LOOSE to ensure trades):
- Trend regime: 4h HMA bullish + 12h HMA bullish → long (no RSI filter)
- Range regime: RSI<35 + price>SMA200 → long, RSI>65 + price<SMA200 → short
- Size: 0.28 (discrete, slightly lower to account for more trades)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3, trades>40/symbol train, >5/symbol test, DD>-35%
Timeframe: 4h (target 30-60 trades/year with dual regime)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_dual_regime_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response with less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    wma_half = close_series.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    High CHOP (>61.8) = range/choppy
    Low CHOP (<38.2) = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """RSI - momentum oscillator"""
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
    """Average True Range - for stoploss"""
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
    """Simple Moving Average - for trend filter"""
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for HTF trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size (slightly lower for more trades)
    
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === CHOPPINESS REGIME DETECTION ===
        is_trend_regime = chop[i] < 45.0  # Low chop = trending
        is_range_regime = chop[i] > 55.0  # High chop = ranging
        # 45-55 is transition zone
        
        # === TREND REGIME LOGIC ===
        trend_signal = 0.0
        if is_trend_regime:
            hma_4h_bull = close[i] > hma_4h[i]
            hma_4h_bear = close[i] < hma_4h[i]
            hma_12h_bull = close[i] > hma_12h_aligned[i]
            hma_12h_bear = close[i] < hma_12h_aligned[i]
            
            # Long: Both HMA bullish
            if hma_4h_bull and hma_12h_bull:
                trend_signal = SIZE
            # Short: Both HMA bearish
            elif hma_4h_bear and hma_12h_bear:
                trend_signal = -SIZE
        
        # === RANGE REGIME LOGIC ===
        range_signal = 0.0
        if is_range_regime:
            sma_ok = not np.isnan(sma_200[i])
            
            # Long: RSI oversold + above SMA200 (uptrend pullback)
            if rsi[i] < 35.0 and sma_ok and close[i] > sma_200[i]:
                range_signal = SIZE
            # Short: RSI overbought + below SMA200 (downtrend rally)
            elif rsi[i] > 65.0 and sma_ok and close[i] < sma_200[i]:
                range_signal = -SIZE
        
        # === COMBINE SIGNALS (priority: range > trend) ===
        desired_signal = 0.0
        if is_range_regime:
            desired_signal = range_signal
        elif is_trend_regime:
            desired_signal = trend_signal
        # else: transition zone, stay flat
        
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