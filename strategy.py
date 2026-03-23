#!/usr/bin/env python3
"""
Experiment #1121: 4h Primary + 1d/1w HTF — Simplified Triple HMA Trend with RSI Pullback

Hypothesis: After analyzing 800+ failed experiments, the key insight is:
1. Complex regime-switching (Choppiness + CRSI + multiple filters) = 0 trades or negative Sharpe
2. SIMPLER works better: Weekly HMA for macro + Daily HMA for trend + 4h RSI for entries
3. Remove conflicting filters that cause 0 trades (experiments #1115, #1118 had Sharpe=0.000)
4. Use Choppiness as ONE simple filter (not regime-switching) to reduce trades in chop
5. Conservative position sizing (0.25-0.30) with 2.5x ATR trailing stop

Why this should beat Sharpe=0.612:
- Weekly HMA provides strongest macro filter (slower = fewer false signals)
- Daily HMA confirms trend alignment (both HTF must agree)
- 4h RSI pullback entries = better entry timing than pure breakout
- Choppiness > 55 = skip trades (simple, not complex regime switching)
- Tested pattern: HMA + RSI + ATR worked on SOL (Sharpe +0.879 in research)

Timeframe: 4h (primary)
HTF: 1d and 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 20-50 trades/year, Sharpe > 0.612, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_triple_hma_rsi_chop_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # 4h HMA for local trend
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.14
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_4h[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA) — Strongest filter ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM TREND (1d HMA) — Must align with weekly ===
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (4h HMA) — Entry timing ===
        local_bull = close[i] > hma_4h[i]
        local_bear = close[i] < hma_4h[i]
        
        # === CHOPPINESS FILTER — Skip trades in choppy market ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === RSI PULLBACK SIGNALS ===
        # Looser thresholds to ensure adequate trade frequency
        rsi_oversold = rsi_4h[i] < 45.0
        rsi_overbought = rsi_4h[i] > 55.0
        rsi_neutral = 35.0 < rsi_4h[i] < 65.0
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # Reduce size in choppy market
        if is_choppy:
            current_size = REDUCED_SIZE
        
        # === LONG ENTRY ===
        # Weekly bull + Daily bull + RSI pullback + (trending OR local bull)
        if macro_bull and trend_bull and rsi_oversold:
            if is_trending or local_bull:
                desired_signal = current_size
        
        # === SHORT ENTRY ===
        # Weekly bear + Daily bear + RSI pullback + (trending OR local bear)
        elif macro_bear and trend_bear and rsi_overbought:
            if is_trending or local_bear:
                desired_signal = -current_size
        
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
                # Hold long if weekly/daily still bull
                if macro_bull and trend_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if weekly/daily still bear
                if macro_bear and trend_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses or RSI very overbought
            if macro_bear or rsi_4h[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses or RSI very oversold
            if macro_bull or rsi_4h[i] < 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = 0.0
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = 0.0
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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