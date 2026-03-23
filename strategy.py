#!/usr/bin/env python3
"""
Experiment #649: 4h Primary + 1d HTF — Supertrend + RSI Pullback + HMA Filter

Hypothesis: 4h Supertrend provides clear trend direction with fewer whipsaws than EMA.
RSI pullback entries (not extremes) catch trend continuations better than breakouts.
1d HMA filter prevents counter-trend trades during major reversals (2022 crash, 2025 bear).

Key innovations:
1. Supertrend(10, 3.0) — clear trend direction, documented edge in crypto
2. RSI(14) pullback entries — long when RSI 40-50 in uptrend, short when 50-60 in downtrend
3. 1d HMA(21) macro filter — only long when price > 1d HMA, only short when price < 1d HMA
4. ATR(14) trailing stop — 2.5x ATR from highest/lowest since entry
5. LOOSE entry thresholds to ensure 30+ trades/year (learned from 0-trade failures)

Why this should beat Sharpe=0.612:
- Supertrend has cleaner signals than EMA crossover (fewer whipsaws)
- RSI pullback (not extreme) entries catch trend continuations
- 1d HTF filter prevents deadly counter-trend trades in 2022/2025
- Conservative sizing (0.30) survives 77% crash with ~27% DD
- Simpler logic = more trades (avoiding the #1 failure: 0 trades)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_rsi_pullback_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator.
    Returns: supertrend_line, trend_direction (1=bullish, -1=bearish)
    
    Formula:
    1. ATR(period)
    2. Upper Band = (High + Low) / 2 + multiplier * ATR
    3. Lower Band = (High + Low) / 2 - multiplier * ATR
    4. Trend flips when price crosses the band
    """
    n = len(close)
    supertrend = np.full(n, np.nan)
    trend = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    if n < period:
        return supertrend, trend
    
    atr = calculate_atr(high, low, close, period)
    
    # Basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[period] = upper_band[period]
    trend[period] = -1 if close[period] < supertrend[period] else 1
    
    # Calculate Supertrend with trend logic
    for i in range(period + 1, n):
        if trend[i-1] == 1:
            # Previously bullish
            if lower_band[i] > supertrend[i-1]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = supertrend[i-1]
            
            if close[i] < supertrend[i]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = 1
        else:
            # Previously bearish
            if upper_band[i] < supertrend[i-1]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i-1]
            
            if close[i] > supertrend[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
    
    return supertrend, trend

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    supertrend_4h, trend_4h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(supertrend_4h[i]) or np.isnan(trend_4h[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === SUPERTREND TREND (4h) ===
        st_bullish = trend_4h[i] == 1
        st_bearish = trend_4h[i] == -1
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 40-55 in uptrend (not oversold, just pullback)
        rsi_long_pullback = 40.0 <= rsi_4h[i] <= 55.0
        # Short: RSI rallied to 45-60 in downtrend (not overbought, just bounce)
        rsi_short_pullback = 45.0 <= rsi_4h[i] <= 60.0
        
        # === RSI MOMENTUM (for entry timing) ===
        # RSI crossing above 45 from below (long momentum)
        rsi_long_momentum = (rsi_4h[i] > 45.0) and (rsi_4h[i-1] <= 45.0) if i > 0 else False
        # RSI crossing below 55 from above (short momentum)
        rsi_short_momentum = (rsi_4h[i] < 55.0) and (rsi_4h[i-1] >= 55.0) if i > 0 else False
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Condition 1: HTF bullish + Supertrend bullish + RSI pullback
        if htf_bullish and st_bullish and rsi_long_pullback:
            desired_signal = SIZE
        # Condition 2: HTF bullish + Supertrend JUST turned bullish + RSI momentum
        elif htf_bullish and st_bullish and trend_4h[i-1] == -1 and rsi_long_momentum:
            desired_signal = SIZE
        # Condition 3: Both HTF and ST bullish + RSI not overbought (hold/add)
        elif htf_bullish and st_bullish and rsi_4h[i] < 70.0:
            desired_signal = SIZE
        
        # === SHORT ENTRY ===
        # Condition 1: HTF bearish + Supertrend bearish + RSI pullback
        elif htf_bearish and st_bearish and rsi_short_pullback:
            desired_signal = -SIZE
        # Condition 2: HTF bearish + Supertrend JUST turned bearish + RSI momentum
        elif htf_bearish and st_bearish and trend_4h[i-1] == 1 and rsi_short_momentum:
            desired_signal = -SIZE
        # Condition 3: Both HTF and ST bearish + RSI not oversold (hold/add)
        elif htf_bearish and st_bearish and rsi_4h[i] > 30.0:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if Supertrend turns bearish
        if in_position and position_side > 0 and st_bearish:
            desired_signal = 0.0
        
        # Exit short if Supertrend turns bullish
        if in_position and position_side < 0 and st_bullish:
            desired_signal = 0.0
        
        # === HTF REVERSAL EXIT ===
        # Exit long if 1d HMA turns bearish
        if in_position and position_side > 0 and htf_bearish:
            desired_signal = 0.0
        
        # Exit short if 1d HMA turns bullish
        if in_position and position_side < 0 and htf_bullish:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
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
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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