#!/usr/bin/env python3
"""
Experiment #006: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Complex regime-switching failed (exp #002, #004). Simpler MTF confluence
should work better. Use 1d HMA for macro bias, 12h HMA for primary trend, RSI for
pullback timing. This proven pattern (from knowledge base) should generate 30-60
trades/year with positive Sharpe.

Key insight from failures:
- Complex regime logic = too many conflicting filters = 0 trades or negative Sharpe
- SIMPLER entry conditions = more trades = better statistics
- RSI pullback (40-60 range) instead of extremes (25/75) = more entry opportunities

Strategy logic:
1. 1d HMA(21): Macro trend bias (only trade with daily trend)
2. 12h HMA(16/48): Primary trend direction
3. RSI(14): Pullback entries (long at 40-50, short at 50-60)
4. ATR(14): 2.5x trailing stoploss
5. Position size: 0.30 (discrete)

Why this should beat previous attempts:
- Fewer conflicting filters = more trades generated
- RSI pullback (not extremes) = catches more moves
- Dual HMA confluence = strong trend filter without overfitting
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

def calculate_hma_fast_slow(close, fast=16, slow=48):
    """Calculate dual HMA for trend confirmation."""
    hma_fast = calculate_hma(close, fast)
    hma_slow = calculate_hma(close, slow)
    return hma_fast, hma_slow

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
    rsi_14 = calculate_rsi(close, period=14)
    hma_12h_fast, hma_12h_slow = calculate_hma_fast_slow(close, fast=16, slow=48)
    
    # Calculate HMA slopes for trend strength
    hma_12h_slope = np.zeros(n)
    for i in range(10, n):
        hma_12h_slope[i] = hma_12h_fast[i] - hma_12h_fast[i-10]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_fast[i]) or np.isnan(hma_12h_slow[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H TREND (Dual HMA) ===
        hma_bullish = hma_12h_fast[i] > hma_12h_slow[i] and hma_12h_slope[i] > 0
        hma_bearish = hma_12h_fast[i] < hma_12h_slow[i] and hma_12h_slope[i] < 0
        
        # === RSI PULLBACK (LOOSE thresholds for trade generation) ===
        rsi_pullback_long = 38.0 < rsi_14[i] < 55.0  # Pullback in uptrend
        rsi_pullback_short = 45.0 < rsi_14[i] < 62.0  # Rally in downtrend
        
        # === RSI MOMENTUM ===
        rsi_rising = rsi_14[i] > rsi_14[i-3] if i >= 3 else False
        rsi_falling = rsi_14[i] < rsi_14[i-3] if i >= 3 else False
        
        # === PRICE POSITION ===
        price_above_hma_fast = close[i] > hma_12h_fast[i]
        price_below_hma_fast = close[i] < hma_12h_fast[i]
        
        # === ENTRY LOGIC (SIMPLE - fewer filters = more trades) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Condition: 1d bullish + 12h bullish + RSI pullback + momentum turning up
        if price_above_hma_1d and hma_bullish:
            if rsi_pullback_long and rsi_rising:
                # Additional confirmation: price above fast HMA
                if price_above_hma_fast:
                    new_signal = POSITION_SIZE
            # Alternative: RSI crossed up from oversold
            elif rsi_14[i] > 40.0 and rsi_14[i-1] <= 40.0:
                if price_above_hma_fast:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Condition: 1d bearish + 12h bearish + RSI rally + momentum turning down
        elif price_below_hma_1d and hma_bearish:
            if rsi_pullback_short and rsi_falling:
                # Additional confirmation: price below fast HMA
                if price_below_hma_fast:
                    new_signal = -POSITION_SIZE
            # Alternative: RSI crossed down from overbought
            elif rsi_14[i] < 60.0 and rsi_14[i-1] >= 60.0:
                if price_below_hma_fast:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        # Exit long if 1d trend turns bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_bearish:
                new_signal = 0.0
        
        # Exit short if 1d trend turns bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_bullish:
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