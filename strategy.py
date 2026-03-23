#!/usr/bin/env python3
"""
Experiment #134: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Choppiness Filter

Hypothesis: Complex regime detection (CRSI+Chop+Donchian) adds lag and reduces trades.
Simpler HMA trend-following with RSI pullback entries has proven success on 4h timeframe.
This strategy uses:

1) 12h HMA(21) for macro trend bias — only trade in trend direction
2) 4h HMA(16/48) crossover for intermediate trend confirmation
3) RSI(14) pullback entries — enter on dips in uptrend (RSI 35-50), rallies in downtrend (RSI 50-65)
4) Choppiness Index(14) filter — avoid trading when CHOP > 61.8 (range-bound)
5) ATR(14) trailing stop at 2.5x — protects capital in reversals

Why this should work:
- HMA is faster than EMA, reduces lag in trend detection
- RSI pullback (not extreme) entries catch continuations, not reversals
- Choppiness filter avoids whipsaws in ranging markets (2023-2024)
- 4h naturally produces 30-50 trades/year (acceptable fee drag)
- Simpler logic = more robust across BTC/ETH/SOL

Position size: 0.25 base, 0.30 with strong confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_chop_12h_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    High CHOP (>61.8) = choppy/ranging, Low CHOP (<38.2) = trending
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for macro trend
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === HTF TREND BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND FILTER (HMA crossover) ===
        hma_4h_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_4h_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === CHOPPINESS FILTER ===
        # Only trade when market is trending (CHOP < 61.8)
        is_trending = chop_14[i] < 61.8
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 35-50 in uptrend
        # Short: RSI rallied to 50-65 in downtrend
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 12h trend up + 4h trend up + trending market + RSI pullback
        if price_above_hma_12h and hma_4h_bullish and is_trending:
            if rsi_pullback_long:
                new_signal = POSITION_SIZE_BASE
                # Increase size if RSI is deeper in pullback (stronger opportunity)
                if rsi_14[i] <= 45.0:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Require: 12h trend down + 4h trend down + trending market + RSI pullback
        if price_below_hma_12h and hma_4h_bearish and is_trending:
            if rsi_pullback_short:
                new_signal = -POSITION_SIZE_BASE
                # Increase size if RSI is higher in rally (stronger opportunity)
                if rsi_14[i] >= 55.0:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold long if trend still intact (don't exit on every RSI fluctuation)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold if 4h HMA still bullish (allow some pullback)
                if hma_4h_bullish and price_above_hma_12h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold if 4h HMA still bearish
                if hma_4h_bearish and price_below_hma_12h:
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit long if 12h trend reverses
            if price_below_hma_12h:
                new_signal = 0.0
            # Exit if 4h HMA crosses bearish
            if hma_4h_bearish:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 12h trend reverses
            if price_above_hma_12h:
                new_signal = 0.0
            # Exit if 4h HMA crosses bullish
            if hma_4h_bullish:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
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
                # Position flip
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