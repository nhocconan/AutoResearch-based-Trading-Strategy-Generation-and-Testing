#!/usr/bin/env python3
"""
Experiment #270: 1h Primary + 4h/12h HTF — Simplified Trend Pullback

Hypothesis: Previous 1h strategies failed from over-filtering (#260: 0 trades, #265: Sharpe=-2.156).
This version uses SIMPLER entry logic:
- 4h HMA(16/48) for PRIMARY trend direction (most important)
- 12h HMA(21) for MACRO bias (soft filter only)
- 1h RSI(14) pullback entries (40-60 zone - triggers frequently)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25 (conservative for 1h volatility)

KEY CHANGES from failed #260:
- REMOVED session filter (was killing trades)
- REMOVED volume filter (was too strict)
- Only 2 HTF filters instead of 3+
- RSI 40-60 triggers ~25% of bars vs 35-50 at ~15%
- 12h HMA is soft bias, not hard requirement

TARGET: 50-100 trades/year on 1h, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_atr_v2"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, 48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 1h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (12h HMA) - SOFT FILTER ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND (HMA crossover) - PRIMARY FILTER ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 1h TREND (HMA crossover) - ENTRY TIMING ===
        hma_1h_bullish = hma_16[i] > hma_48[i]
        hma_1h_bearish = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK SIGNALS (40-60 zone triggers frequently) ===
        rsi_pullback_long = (rsi_14[i] >= 40.0) and (rsi_14[i] <= 60.0)
        rsi_pullback_short = (rsi_14[i] >= 40.0) and (rsi_14[i] <= 60.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bullish + 1h bullish + RSI pullback + 12h bias (soft)
        if hma_4h_bullish and hma_1h_bullish and rsi_pullback_long:
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 4h bearish + 1h bearish + RSI pullback + 12h bias (soft)
        elif hma_4h_bearish and hma_1h_bearish and rsi_pullback_short:
            desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_4h_bullish:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and hma_4h_bearish:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals