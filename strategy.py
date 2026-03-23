#!/usr/bin/env python3
"""
Experiment #007: 1d Primary + 1w HTF — HMA Trend + RSI Pullback + ATR Stop

Hypothesis: After 6 failed experiments with complex regime filters, I'm returning to
a proven simple pattern that worked on SOL (Sharpe +0.879 in research): HMA crossover
with RSI filter for entry timing and ATR trailing stop for risk management.

Key differences from failed attempts:
1. NO Choppiness Index (failed in #001, #002, #006)
2. NO Connors RSI complexity (failed in #001, #006)
3. NO Donchian breakouts (failed in #002)
4. NO Fisher Transform (failed in #003 - 0 trades)
5. NO complex regime switching (failed in #005 - 0 trades)
6. NO vol spike confluence (failed in #004 - Sharpe -11.9)

Why this might work:
- HMA is faster than EMA, catches trends earlier (research-backed)
- RSI filter prevents entering at extremes (reduces whipsaw)
- 1w HTF provides major trend bias (avoid fighting macro trend)
- 1d TF targets 20-50 trades/year (fee-efficient per Rule 10)
- ATR trailing stop protects capital (mandatory per Rule 6)
- Position size 0.30 (conservative for 1d per Rule 4)

Entry conditions (LOOSE enough to generate trades on ALL symbols):
- Long: 1d HMA crosses above 1d HMA(48) OR price > 1w HMA, AND RSI(14) < 60
- Short: 1d HMA crosses below 1d HMA(48) OR price < 1w HMA, AND RSI(14) > 40
- Either HMA crossover OR price vs 1w HMA triggers (not both required)

Stoploss: 2.5*ATR trailing, signal→0 when hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_v1"
timeframe = "1d"
leverage = 1.0

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
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_1d_fast = calculate_hma(close, period=21)
    hma_1d_slow = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_fast[i]):
            continue
        if np.isnan(hma_1d_slow[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(rsi_14[i]) or atr_14[i] == 0:
            continue
        
        # === 1W MAJOR TREND BIAS ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_aligned[i]
        
        # === 1D HMA CROSSOVER ===
        hma_cross_bull = hma_1d_fast[i] > hma_1d_slow[i]
        hma_cross_bear = hma_1d_fast[i] < hma_1d_slow[i]
        
        # Check previous bar for crossover detection
        hma_cross_bull_prev = hma_1d_fast[i-1] > hma_1d_slow[i-1] if i > 0 else False
        hma_cross_bear_prev = hma_1d_fast[i-1] < hma_1d_slow[i-1] if i > 0 else False
        
        # Fresh crossover (just crossed)
        fresh_bull_cross = hma_cross_bull and not hma_cross_bull_prev
        fresh_bear_cross = hma_cross_bear and not hma_cross_bear_prev
        
        # === RSI FILTER ===
        rsi_neutral_long = rsi_14[i] < 65  # Not overbought
        rsi_neutral_short = rsi_14[i] > 35  # Not oversold
        
        # === ENTRY LOGIC (LOOSE - either condition works) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Condition 1: Fresh HMA bull crossover + RSI not overbought
        long_cross = fresh_bull_cross and rsi_neutral_long
        
        # Condition 2: Price above 1w HMA (major trend) + RSI pullback
        long_trend = price_above_1w_hma and rsi_14[i] < 55
        
        # Condition 3: HMA already bullish + RSI oversold (pullback entry)
        long_pullback = hma_cross_bull and rsi_14[i] < 40
        
        if long_cross or long_trend or long_pullback:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Condition 1: Fresh HMA bear crossover + RSI not oversold
        short_cross = fresh_bear_cross and rsi_neutral_short
        
        # Condition 2: Price below 1w HMA (major trend) + RSI rally
        short_trend = price_below_1w_hma and rsi_14[i] > 45
        
        # Condition 3: HMA already bearish + RSI overbought (rally entry)
        short_pullback = hma_cross_bear and rsi_14[i] > 60
        
        if short_cross or short_trend or short_pullback:
            # Only short if not already in long
            if new_signal <= 0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if HMA turns bearish
            if hma_cross_bear and price_below_1w_hma:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HMA turns bullish
            if hma_cross_bull and price_above_1w_hma:
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
                # Flip position
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