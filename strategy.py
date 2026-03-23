#!/usr/bin/env python3
"""
Experiment #404: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback + Z-Score

Hypothesis: The complex regime-switching in #399 caused 0 trades / negative Sharpe.
Returning to proven pattern from mtf_hma_rsi_zscore_v1 (Sharpe=5.4 baseline):
1. 4h HMA(16/48) for primary trend direction
2. 12h HMA(21) for intermediate bias filter
3. 1d HMA(21) for long-term bias filter
4. RSI(7) pullback entries in trend direction (simpler than CRSI)
5. Z-score(20) for extreme mean-reversion entries
6. ATR(14) trailing stoploss (2.5x)
7. Discrete position sizing: 0.0, ±0.25, ±0.30

Why this should beat Sharpe=0.612:
- Simpler entry logic = MORE trades (fixes #399's 0-trade problem)
- HMA trend + RSI pullback = proven combination (Sharpe=5.4 baseline)
- Multi-HTF alignment (12h + 1d) = stronger bias filter than single HTF
- Z-score adds mean-reversion edge in ranging markets
- 4h TF targets 30-50 trades/year = acceptable fee drag (~2%)

Key fix vs #399: Remove Choppiness Index (too many filters = no trades).
Use simpler HMA slope + RSI + Z-score confluence.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_zscore_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (close_s - rolling_mean) / (rolling_std + 1e-10)
    
    return zscore.values

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)."""
    n = len(hma)
    slope = np.zeros(n)
    for i in range(lookback, n):
        slope[i] = (hma[i] - hma[i - lookback]) / (hma[i - lookback] + 1e-10)
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    hma_16_slope = calculate_hma_slope(hma_16, lookback=5)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    zscore_20 = calculate_zscore(close, period=20)
    
    # Calculate and align HTF HMA for bias (12h and 1d)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h (target 30-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(zscore_20[i]):
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(hma_16_slope[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === PRIMARY TREND (4h HMA crossover + slope) ===
        hma_bullish = hma_16[i] > hma_48[i] and hma_16_slope[i] > 0.001
        hma_bearish = hma_16[i] < hma_48[i] and hma_16_slope[i] < -0.001
        
        # === HTF BIAS (12h and 1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_neutral = 35.0 <= rsi_7[i] <= 65.0
        
        # === Z-SCORE EXTREMES ===
        zscore_extreme_low = zscore_20[i] < -1.5
        zscore_extreme_high = zscore_20[i] > 1.5
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Multiple paths (looser conditions for more trades)
        long_bias = price_above_hma_12h or price_above_hma_1d  # At least one HTF bullish
        
        if long_bias:
            if hma_bullish and rsi_oversold:
                # Trend + RSI pullback (primary entry)
                desired_signal = BASE_SIZE
            elif hma_bullish and zscore_extreme_low:
                # Trend + Z-score mean reversion
                desired_signal = BASE_SIZE
            elif hma_bullish and rsi_neutral:
                # Trend continuation (no pullback needed)
                desired_signal = BASE_SIZE * 0.5  # Smaller size for continuation
        
        # SHORT SETUP - Multiple paths
        short_bias = price_below_hma_12h or price_below_hma_1d  # At least one HTF bearish
        
        if short_bias:
            if hma_bearish and rsi_overbought:
                # Trend + RSI pullback (primary entry)
                desired_signal = -BASE_SIZE
            elif hma_bearish and zscore_extreme_high:
                # Trend + Z-score mean reversion
                desired_signal = -BASE_SIZE
            elif hma_bearish and rsi_neutral:
                # Trend continuation
                desired_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS CHECK (ATR trailing) ===
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
        
        # === RSI EXIT (extreme reached) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_1d and price_below_hma_12h:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d and price_above_hma_12h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or price_above_hma_12h):
                desired_signal = BASE_SIZE * 0.5  # Hold with reduced size
            elif position_side < 0 and (hma_bearish or price_below_hma_12h):
                desired_signal = -BASE_SIZE * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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