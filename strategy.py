#!/usr/bin/env python3
"""
Experiment #014: 4h Primary + 12h HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 13 failed experiments with overly complex confluence filters (many got 0 trades),
I'm simplifying to proven patterns: HMA trend direction + RSI pullback entries.

Key learnings from failures:
- #004 vol spike: Sharpe=-11.969 (too many false signals)
- #006 CRSI+Chop: Sharpe=-0.136 (overly complex)
- #010 CRSI+Chop 1h: Sharpe=-1.796 (lower TF = fee drag)
- #011 Fisher+HMA: Sharpe=-0.835 (Fisher too noisy)
- Many strategies: Sharpe=0.000 = 0 TRADES (entry conditions too strict!)

Why this might work:
1. SIMPLER entry logic = more trades (avoid 0-trade failure)
2. 12h HMA for trend bias (proven in research)
3. RSI 4h for pullback timing (loose thresholds: 35/65 not 30/70)
4. ADX filter to avoid dead chop (ADX > 18, not > 25)
5. Position size 0.30 (discrete, conservative)
6. ATR stoploss 2.5x (mandatory per rules)

Entry conditions (LOOSE to ensure trades):
- Long: 12h HMA bullish + 4h RSI < 40 + ADX > 18
- Short: 12h HMA bearish + 4h RSI > 60 + ADX > 18

Exit: RSI mean reversion (cross 50) OR stoploss hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_trend_rsi_pullback_12h_v2"
timeframe = "4h"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
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
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # 4h HMA for additional trend confirmation
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(hma_4h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3] if i >= 3 else False
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-3] if i >= 3 else False
        price_above_hma_4h = close[i] > hma_4h[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        
        # === ADX FILTER (avoid dead chop) ===
        adx_trending = adx_14[i] > 18  # Loose threshold to allow trades
        
        # === RSI PULLBACK ENTRY (LOOSE thresholds) ===
        rsi_oversold = rsi_14[i] < 40  # Not 30 - too strict
        rsi_overbought = rsi_14[i] > 60  # Not 70 - too strict
        
        # === ENTRY LOGIC (SIMPLIFIED) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # 12h trend bullish + 4h RSI pullback + ADX confirms trend
        long_condition = (
            (hma_12h_slope_bull or price_above_hma_12h) and  # 12h bias
            rsi_oversold and  # 4h pullback
            adx_trending  # Not dead chop
        )
        
        if long_condition:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # 12h trend bearish + 4h RSI pullback + ADX confirms trend
        short_condition = (
            (hma_12h_slope_bear or price_below_hma_12h) and  # 12h bias
            rsi_overbought and  # 4h pullback
            adx_trending  # Not dead chop
        )
        
        if short_condition:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Check for exit conditions
            exit_signal = False
            
            # Exit long on RSI mean reversion
            if position_side > 0 and rsi_14[i] > 55:
                exit_signal = True
            
            # Exit short on RSI mean reversion
            if position_side < 0 and rsi_14[i] < 45:
                exit_signal = True
            
            # Exit on trend flip
            if position_side > 0 and hma_12h_slope_bear and price_below_hma_12h:
                exit_signal = True
            
            if position_side < 0 and hma_12h_slope_bull and price_above_hma_12h:
                exit_signal = True
            
            if exit_signal:
                new_signal = 0.0
            else:
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