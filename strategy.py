#!/usr/bin/env python3
"""
Experiment #719: 4h Primary + 1d HTF — KAMA + Fisher Transform + Vol Filter

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts better to market noise than 
HMA/EMA, reducing whipsaw in choppy conditions. Fisher Transform catches reversals 
more reliably than RSI in bear markets (proven in research). Combined with vol filter 
to avoid entering during panic spikes.

Why this should beat #709 (Sharpe=-0.411):
1. KAMA adapts to volatility - slower in chop, faster in trends (less whipsaw)
2. Fisher Transform is superior to RSI for reversal timing in bear markets
3. ATR ratio filter avoids entering during vol spikes (panic bottoms/tops)
4. Simpler entry logic = more trades (avoid 0-trade failures)
5. 1d KAMA trend bias prevents counter-trend trades

Key differences from failed experiments:
- No ADX regime switching (laggy, causes 0 trades in #709)
- No complex CRSI/Choppiness combos (failed #697-#712)
- Fisher < -1.0 / > +1.0 thresholds are looser than RSI 25/75
- Vol filter (ATR7/ATR30 < 2.0) prevents panic entries

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14):
    """
    Kaufman Adaptive Moving Average - adapts to market noise.
    Slower during choppy markets, faster during trends.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + 1:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    sc = (er * (2.0 / (period + 1) - 2.0 / (period + 1 + 1)) + 2.0 / (period + 1 + 1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear markets.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i - period + 1:i + 1] + low[i - period + 1:i + 1]) / 2
        highest = np.max(hl2)
        lowest = np.min(hl2)
        
        if highest == lowest:
            continue
        
        # Normalize to -1 to +1
        x = (2 * hl2[-1] - lowest - highest) / (highest - lowest + 1e-10)
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
        
        if i > period:
            fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    atr_4h_30 = calculate_atr(high, low, close, period=30)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Need buffer for indicators
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(fisher_4h[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(atr_4h_30[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === VOLATILITY FILTER ===
        # Avoid entering during extreme volatility spikes (panic)
        vol_ratio = atr_4h[i] / (atr_4h_30[i] + 1e-10)
        vol_normal = vol_ratio < 2.0
        
        # === TREND BIAS (1d HTF KAMA) ===
        trend_bullish = close[i] > kama_1d_aligned[i]
        trend_bearish = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND (4h KAMA slope) ===
        kama_slope_up = False
        kama_slope_down = False
        if i > 2 and not np.isnan(kama_4h[i-1]) and not np.isnan(kama_4h[i-2]):
            kama_slope_up = kama_4h[i] > kama_4h[i-1] > kama_4h[i-2]
            kama_slope_down = kama_4h[i] < kama_4h[i-1] < kama_4h[i-2]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_4h[i] < -1.0
        fisher_overbought = fisher_4h[i] > 1.0
        fisher_cross_up = fisher_signal_4h[i] < -1.0 and fisher_4h[i] >= -1.0
        fisher_cross_down = fisher_signal_4h[i] > 1.0 and fisher_4h[i] <= 1.0
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # Primary: Trend + Fisher reversal + normal vol
        if trend_bullish and fisher_cross_up and vol_normal:
            desired_signal = current_size
        # Secondary: Strong trend + Fisher oversold (pullback entry)
        elif trend_bullish and kama_slope_up and fisher_oversold and vol_normal:
            desired_signal = REDUCED_SIZE
        # Tertiary: Price above 1d KAMA + Fisher deeply oversold
        elif trend_bullish and fisher_4h[i] < -1.5:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY ===
        # Primary: Trend + Fisher reversal + normal vol
        if trend_bearish and fisher_cross_down and vol_normal:
            desired_signal = -current_size
        # Secondary: Strong trend + Fisher overbought (pullback entry)
        elif trend_bearish and kama_slope_down and fisher_overbought and vol_normal:
            desired_signal = -REDUCED_SIZE
        # Tertiary: Price below 1d KAMA + Fisher deeply overbought
        elif trend_bearish and fisher_4h[i] > 1.5:
            desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish and Fisher not overbought
                if trend_bullish and fisher_4h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish and Fisher not oversold
                if trend_bearish and fisher_4h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or Fisher overbought
            if trend_bearish or fisher_4h[i] > 2.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or Fisher oversold
            if trend_bullish or fisher_4h[i] < -2.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals