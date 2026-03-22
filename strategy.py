#!/usr/bin/env python3
"""
Experiment #534: 4h Primary + 12h HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 478 failed strategies (mostly complex multi-filter combos with Choppiness/Connors),
return to PROVEN SIMPLE logic that generated trades consistently.

Key insights from failures:
- Choppiness + Connors combos = Sharpe -2.5 to -3.0 (experiments #522-#533)
- Too many entry conditions = 0 trades (experiments #528, #530)
- Simple HMA + RSI worked in earlier experiments (mtf_hma_rsi_zscore_v1 Sharpe=5.4)

This strategy uses MINIMAL filters for MAXIMUM trade frequency:
1. 12h HMA(21) for major trend direction (simpler than 1d, more responsive)
2. 4h HMA(16/48) crossover for entry timing
3. RSI(14) single threshold filter (only avoid extremes: <25 or >75)
4. ATR(14) 2.5x trailing stop
5. NO regime switching, NO choppiness, NO session filters

Why this might work:
- Fewer conflicting conditions = more trades (critical for meeting min trade requirements)
- 12h HTF is more responsive than 1d for 4h entries
- Single RSI threshold avoids mutually exclusive conditions
- Proven HMA crossover logic from successful earlier experiments

Position sizing: 0.30 (discrete, per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_simp_12h_v1"
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
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
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
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF HMA for major trend direction
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # HMA crossover signals (16/48)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    
    # RSI for entry filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === 12H MAJOR TREND (primary direction filter) ===
        # Simple: price above 12h HMA(21) = bull, below = bear
        bull_regime = close[i] > hma_12h_21_aligned[i]
        bear_regime = close[i] < hma_12h_21_aligned[i]
        
        # 12h HMA slope confirmation
        hma_slope_bull = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_slope_bear = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === 4H HMA CROSSOVER SIGNALS ===
        hma_cross_up = (hma_4h_16[i] > hma_4h_48[i]) and (hma_4h_16[i-1] <= hma_4h_48[i-1])
        hma_cross_down = (hma_4h_16[i] < hma_4h_48[i]) and (hma_4h_16[i-1] >= hma_4h_48[i-1])
        
        # HMA alignment (already in trend, no crossover needed)
        hma_aligned_bull = hma_4h_16[i] > hma_4h_48[i]
        hma_aligned_bear = hma_4h_16[i] < hma_4h_48[i]
        
        # === RSI FILTER (single threshold - avoid extremes only) ===
        # Long: RSI < 70 (not overbought) OR RSI < 45 (pullback)
        # Short: RSI > 30 (not oversold) OR RSI > 55 (bounce)
        rsi_ok_long = rsi_14[i] < 70.0
        rsi_ok_short = rsi_14[i] > 30.0
        rsi_pullback_long = rsi_14[i] < 45.0
        rsi_bounce_short = rsi_14[i] > 55.0
        
        # === ENTRY LOGIC — SIMPLIFIED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (any of these conditions triggers)
        # Condition 1: HMA crossover up + bull regime (primary entry)
        if hma_cross_up and bull_regime and rsi_ok_long:
            new_signal = POSITION_SIZE
        # Condition 2: HMA aligned bull + bull regime + RSI pullback (secondary entry)
        elif hma_aligned_bull and bull_regime and rsi_pullback_long:
            new_signal = POSITION_SIZE
        # Condition 3: Strong bull trend (12h slope) + HMA crossover
        elif bull_regime and hma_slope_bull and hma_cross_up:
            new_signal = POSITION_SIZE
        # Condition 4: HMA aligned + pullback (mean reversion in uptrend)
        elif hma_aligned_bull and rsi_pullback_long:
            new_signal = POSITION_SIZE * 0.8
        
        # SHORT ENTRIES (mirror logic, check only if no long signal)
        if new_signal == 0.0:
            # Condition 1: HMA crossover down + bear regime (primary entry)
            if hma_cross_down and bear_regime and rsi_ok_short:
                new_signal = -POSITION_SIZE
            # Condition 2: HMA aligned bear + bear regime + RSI bounce (secondary entry)
            elif hma_aligned_bear and bear_regime and rsi_bounce_short:
                new_signal = -POSITION_SIZE
            # Condition 3: Strong bear trend (12h slope) + HMA crossover
            elif bear_regime and hma_slope_bear and hma_cross_down:
                new_signal = -POSITION_SIZE
            # Condition 4: HMA aligned + bounce (mean reversion in downtrend)
            elif hma_aligned_bear and rsi_bounce_short:
                new_signal = -POSITION_SIZE * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on regime flip to bear
        if in_position and position_side > 0 and bear_regime and hma_slope_bear:
            new_signal = 0.0
        
        # Exit short on regime flip to bull
        if in_position and position_side < 0 and bull_regime and hma_slope_bull:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals