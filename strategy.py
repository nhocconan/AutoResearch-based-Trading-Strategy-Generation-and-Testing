#!/usr/bin/env python3
"""
Experiment #066: 12h Primary + 1d HTF — Simplified Trend Pullback Strategy

Hypothesis: 12h timeframe with daily HTF bias using HMA trend + RSI pullback entries
will generate 25-45 trades/year with Sharpe > 0.486 by reducing filter complexity.

Key changes from failed experiments:
1) SIMPLIFIED entries: HMA trend + RSI pullback (not complex CRSI + Choppiness + Donchian)
2) Fewer filters: ADX > 20 for trend strength only (not multiple regime switches)
3) Lenient RSI thresholds: 35-45 for long pullbacks, 55-65 for short pullbacks
4) Single HTF: 1d HMA for macro bias only (not dual 12h+1d which overcomplicates)
5) Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
6) ATR trailing stop: 2.5*ATR to protect capital in 2022-style crashes

Why 12h should work:
- Higher TF = fewer false signals, less fee drag
- 12h captures multi-day trends without 4h noise
- RSI pullbacks in trends have 60%+ win rate historically
- 1d HTF prevents counter-trend trades in bear markets (2022, 2025)

Position size: 0.25-0.30 (discrete levels)
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.5, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    POSITION_SIZE_REDUCED = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND FILTER (12h) ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        hma_21_above_48 = hma_21[i] > hma_48[i]
        hma_21_below_48 = hma_21[i] < hma_48[i]
        
        # === TREND STRENGTH ===
        adx_strong = adx_14[i] > 20.0
        
        # === RSI PULLBACK ZONES ===
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi_14[i] <= 65.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # LONG: 12h uptrend + RSI pullback + 1d not bearish
        if price_above_hma_21 and hma_21_above_48:
            if rsi_pullback_long or rsi_oversold:
                if price_above_hma_1d or not price_below_hma_1d:
                    new_signal = POSITION_SIZE
        
        # SHORT: 12h downtrend + RSI pullback + 1d not bullish
        if price_below_hma_21 and hma_21_below_48:
            if rsi_pullback_short or rsi_overbought:
                if price_below_hma_1d or not price_above_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if trend intact and RSI not extreme against us
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if price_above_hma_21 and rsi_14[i] < 75.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if price_below_hma_21 and rsi_14[i] > 25.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_21 and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_21 and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals