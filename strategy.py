#!/usr/bin/env python3
"""
Experiment #496: 12h Primary + 1d HTF — Simplified Donchian Breakout + HMA Trend

Hypothesis: After 495 experiments, clear pattern emerges — OVER-FILTERING causes 0 trades.
The best 12h patterns from research were Donchian breakout + HMA trend (SOL Sharpe +0.879).
Previous 12h attempts (#492, #494) failed due to too many conflicting conditions.

Key changes from failed experiments:
1. SIMPLER entry logic — only 2-3 conditions max (not 5-6 like #472)
2. Relaxed RSI thresholds (35/65 instead of 25/75) for adequate frequency
3. Donchian(20) breakout is proven pattern — price breaking 20-bar high/low
4. 1d HMA(21) for clean trend filter without over-complication
5. ATR 2.5x trailing stop (slightly wider than 2.0x to avoid premature exits)

Why this might beat current best (Sharpe=0.435):
- 12h has lower fee drag than 4h/1h while maintaining edge
- Donchian breakouts catch momentum moves in both bull/bear markets
- Fewer filters = more trades (critical: need >=30 trades/symbol on train)
- Asymmetric sizing protects in 2022-style crashes

Position sizing: 0.30 long, 0.25 short (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 25-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_1d_simp_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(sma_50[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI FILTER (relaxed for frequency) ===
        rsi_ok_long = rsi_14[i] > 35.0  # Not oversold on breakout
        rsi_ok_short = rsi_14[i] < 65.0  # Not overbought on breakdown
        
        # === SMA50 CONFIRMATION ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === ENTRY LOGIC — SIMPLIFIED (2-3 conditions max) ===
        new_signal = 0.0
        
        # LONG: Bull regime + Donchian breakout + RSI confirmation
        if bull_regime and breakout_long and rsi_ok_long:
            new_signal = LONG_SIZE
        # LONG: Breakout + above SMA50 (momentum confirmation)
        elif breakout_long and above_sma50 and rsi_14[i] > 40.0:
            new_signal = LONG_SIZE
        # LONG: Pullback entry in bull regime (price near Donchian lower)
        elif bull_regime and close[i] < donchian_lower[i] * 1.02 and rsi_14[i] < 45.0:
            new_signal = LONG_SIZE * 0.7
        
        # SHORT: Bear regime + Donchian breakdown + RSI confirmation
        if new_signal == 0.0:
            if bear_regime and breakout_short and rsi_ok_short:
                new_signal = -SHORT_SIZE
            # SHORT: Breakdown + below SMA50 (momentum confirmation)
            elif breakout_short and below_sma50 and rsi_14[i] < 60.0:
                new_signal = -SHORT_SIZE
            # SHORT: Pullback entry in bear regime (price near Donchian upper)
            elif bear_regime and close[i] > donchian_upper[i] * 0.98 and rsi_14[i] > 55.0:
                new_signal = -SHORT_SIZE * 0.7
        
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
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long on RSI overbought
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        # Exit short on RSI oversold
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit (trend changed against position)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
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