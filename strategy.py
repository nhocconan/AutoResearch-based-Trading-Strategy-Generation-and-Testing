#!/usr/bin/env python3
"""
Experiment #1436: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After analyzing 1074 failed strategies, the #1 failure mode is 0 trades (Sharpe=0.000).
Experiment #1432 achieved Sharpe=0.484 with 12h HMA+RSI+Chop+1d trend. This suggests:
1. 12h timeframe CAN work with proper trade frequency
2. Simpler conditions = more trades = better Sharpe
3. Over-filtering (CRSI<15, CHOP>61.8, multiple confluence) = 0 trades

Design (SIMPLIFIED to ensure >=30 trades train, >=3 test):
1. 1d HMA(21) = macro trend filter (call ONCE before loop via mtf_data)
2. 12h HMA(21) = local trend direction
3. RSI(14) pullback entry: long when RSI 35-50 in uptrend, short when RSI 50-65 in downtrend
4. ATR(14) 2.5x trailing stop = risk management
5. Optional: Choppiness Index to reduce trades in extreme chop (but not block all signals)
6. Position size 0.28 = conservative for 12h volatility

Why this works:
- HMA trend filter removes counter-trend trades (major loss source)
- RSI pullback entries catch retracements, not breakouts (higher win rate)
- 12h timeframe = ~30-50 trades/year target (fee-efficient)
- 1d HTF = proper trend alignment without over-complicating

Target: Sharpe > 0.618 (beat current best), trades >= 30 train, >= 3 test, DD < -40%
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # HMA slope tracking for trend confirmation
    hma_slope_prev = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - primary filter ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (12h HMA) ===
        local_bull = close[i] > hma_12h[i]
        local_bear = close[i] < hma_12h[i]
        
        # === HMA SLOPE (trend momentum) ===
        if i > 5 and not np.isnan(hma_12h[i-5]):
            hma_slope = (hma_12h[i] - hma_12h[i-5]) / hma_12h[i-5] if hma_12h[i-5] > 0 else 0.0
        else:
            hma_slope = 0.0
        
        slope_bull = hma_slope > 0.001
        slope_bear = hma_slope < -0.001
        
        # === RSI PULLBACK ENTRY (simplified for trade frequency) ===
        # Long: RSI pulled back to 35-50 in uptrend
        rsi_long_pullback = 35.0 <= rsi[i] <= 55.0
        # Short: RSI pulled back to 50-65 in downtrend
        rsi_short_pullback = 45.0 <= rsi[i] <= 65.0
        
        # === CHOPPINESS FILTER (optional - don't block all signals) ===
        is_extreme_chop = chop[i] > 65.0  # Only avoid extreme chop
        
        # === DESIRED SIGNAL - SIMPLIFIED LOGIC ===
        desired_signal = 0.0
        
        # LONG: Macro bull + local bull + RSI pullback + not extreme chop
        if macro_bull and local_bull and rsi_long_pullback and not is_extreme_chop:
            desired_signal = BASE_SIZE
        # LONG: Macro bull + slope bull + RSI oversold (<40)
        elif macro_bull and slope_bull and rsi[i] < 40.0:
            desired_signal = BASE_SIZE
        # LONG: Strong macro bull (price > 1d HMA by 2%) + RSI < 50
        elif macro_bull and (close[i] > hma_1d_aligned[i] * 1.02) and rsi[i] < 50.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT: Macro bear + local bear + RSI pullback + not extreme chop
        elif macro_bear and local_bear and rsi_short_pullback and not is_extreme_chop:
            desired_signal = -BASE_SIZE
        # SHORT: Macro bear + slope bear + RSI overbought (>60)
        elif macro_bear and slope_bear and rsi[i] > 60.0:
            desired_signal = -BASE_SIZE
        # SHORT: Strong macro bear (price < 1d HMA by 2%) + RSI > 50
        elif macro_bear and (close[i] < hma_1d_aligned[i] * 0.98) and rsi[i] > 50.0:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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