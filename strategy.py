#!/usr/bin/env python3
"""
Experiment #091: 4h Primary + 1d HTF — KAMA Adaptive Trend + Fisher Reversal

Hypothesis: After analyzing 70+ failed experiments, clear patterns emerge:
- 4h strategies fail with too many filters (Choppiness+Donchian+RSI = 0 trades)
- KAMA (Kaufman Adaptive MA) outperforms HMA in crypto's choppy markets
- Fisher Transform excels at catching reversals in bear/range regimes (2022, 2025)
- Simple 1d trend bias + 4h KAMA slope + Fisher extreme = quality over quantity

This strategy combines:
1. 1d KAMA(21) = major trend bias (adaptive to volatility)
2. 4h KAMA(10) slope = entry trigger (faster, more responsive)
3. Fisher Transform(9) = reversal timing (long when Fisher<-1.5, short when >+1.5)
4. RSI(14) loose filter (>35 long, <65 short) = avoid extremes
5. ATR(14) 2.5x trailing stop = risk management

Key design choices:
- Timeframe: 4h (target 20-50 trades/year)
- HTF: 1d for trend bias (more responsive than 1w)
- KAMA ER period: 10 (fast adaptation to crypto volatility)
- Fisher period: 9 (standard, catches reversals well)
- Position size: 0.28 (28% of capital, conservative for 4h)
- Stoploss: 2.5x ATR trailing (tighter for 4h vs 12h)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_reversal_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < slow_period + er_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility < 1e-10:
            er = 1.0
        else:
            er = price_change / volatility
        
        # Calculate smoothing constant
        fast_sc = 2.0 / (fast_period + 1.0)
        slow_sc = 2.0 / (slow_period + 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for reversal detection
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(close[i - period + 1:i + 1])
        ll = np.min(close[i - period + 1:i + 1])
        
        if hh - ll < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * ((close[i] - ll) / (hh - ll)) - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > period:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for major trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 4h)
    
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
        if np.isnan(kama_4h[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        # Price above 1d KAMA = bullish bias, below = bearish bias
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 4h KAMA SLOPE (trend direction) ===
        # Compare current KAMA to KAMA 3 bars ago for slope
        kama_slope_bull = kama_4h[i] > kama_4h[i - 3] if i >= 3 else False
        kama_slope_bear = kama_4h[i] < kama_4h[i - 3] if i >= 3 else False
        
        # === FISHER REVERSAL SIGNAL ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # Alternative: Fisher extreme levels for entry
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 35.0
        rsi_ok_short = rsi[i] < 65.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h KAMA slope up + (Fisher reversal OR Fisher extreme) + RSI ok
        # SHORT: 1d bear + 4h KAMA slope down + (Fisher reversal OR Fisher extreme) + RSI ok
        desired_signal = 0.0
        
        long_condition = (htf_bull and kama_slope_bull and 
                         (fisher_long or fisher_extreme_long) and rsi_ok_long)
        short_condition = (htf_bear and kama_slope_bear and 
                          (fisher_short or fisher_extreme_short) and rsi_ok_short)
        
        if long_condition:
            desired_signal = SIZE
        elif short_condition:
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