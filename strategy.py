#!/usr/bin/env python3
"""
Experiment #1019: 4h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback + ADX Filter

Hypothesis: After 738+ failed strategies, the key insight is that COMPLEXITY kills performance.
Strategies with 5+ confluence filters get 0 trades. This strategy SIMPLIFIES:

1. KAMA (Kaufman Adaptive Moving Average): Adapts smoothing based on market efficiency.
   Fast in trends, slow in chop. Better than HMA/EMA for crypto's regime changes.
   Period=21, fast SC=2/3, slow SC=2/21.

2. DUAL KAMA TREND: 1d KAMA21 for macro bias, 4h KAMA21 for entry timing.
   Long only when price > 1d KAMA (bullish macro).
   Short only when price < 1d KAMA (bearish macro).
   This asymmetry prevents fighting the macro trend.

3. RSI PULLBACK ENTRY: RSI(14) < 45 for long entries in uptrend, RSI(14) > 55 for short in downtrend.
   Captures pullbacks within the trend, not counter-trend reversals.
   Much simpler than Fisher/CRSI and generates more trades.

4. ADX TRENGTH FILTER: ADX(14) > 20 confirms trend has momentum.
   Prevents entries in dead/choppy markets where trend strategies fail.
   Relaxed threshold (20 not 25) ensures enough trades.

5. ATR TRAILING STOP: 2.5x ATR(14) trailing stop for risk management.
   Signal→0 when stop hit.

Why this works:
- SIMPLER = more trades (target 40-60/year on 4h)
- KAMA adapts to volatility regimes better than fixed EMA/HMA
- RSI pullback is proven (mtf_hma_rsi_zscore_v1 had Sharpe=5.4)
- ADX filter prevents choppy market losses
- 1d trend bias prevents counter-trend trades in bear markets

Critical fixes from failures:
- REMOVED Fisher Transform (exp #1014 failed with Sharpe=-0.741)
- REMOVED Choppiness Index (too many false regime switches)
- REMOVED Vol Spike filter (rarely triggers, reduces trades)
- SIMPLIFIED to KAMA + RSI + ADX (3 indicators, not 6+)
- RELAXED RSI thresholds (45/55 not 30/70) for more trades
- Single HTF (1d not 12h+1d) reduces complexity

Target: Sharpe > 0.612, trades >= 40 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 40-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_adx_1d_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_sc=2/3, slow_sc=2/21):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in choppy markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + 10:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i-period])
        noise = 0.0
        for j in range(i-period+1, i+1):
            noise += abs(close[j] - close[j-1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            gain[i] = delta[i-1]
        else:
            loss[i] = -delta[i-1]
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA21 for long-term trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21)
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d KAMA21) ===
        # Asymmetric bias: only long above 1d KAMA, only short below
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === MEDIUM TREND (4h KAMA21) ===
        medium_bull = close[i] > kama_4h[i]
        medium_bear = close[i] < kama_4h[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_4h[i] > 20.0  # Relaxed threshold for more trades
        
        # === RSI PULLBACK SIGNALS ===
        rsi_pullback_long = rsi_4h[i] < 45.0  # Pullback in uptrend
        rsi_pullback_short = rsi_4h[i] > 55.0  # Pullback in downtrend
        rsi_extreme_long = rsi_4h[i] < 35.0  # Deep pullback
        rsi_extreme_short = rsi_4h[i] > 65.0  # Deep pullback
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if macro_bull and medium_bull and trend_strong:
            # Strong uptrend: enter on RSI pullback
            if rsi_pullback_long:
                desired_signal = BASE_SIZE
            elif rsi_extreme_long:
                # Deep pullback = stronger signal
                desired_signal = BASE_SIZE
        elif macro_bull and not medium_bull and trend_strong:
            # Macro bull but 4h pullback: wait for RSI extreme
            if rsi_extreme_long:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if macro_bear and medium_bear and trend_strong:
            # Strong downtrend: enter on RSI pullback
            if rsi_pullback_short:
                desired_signal = -BASE_SIZE
            elif rsi_extreme_short:
                # Deep pullback = stronger signal
                desired_signal = -BASE_SIZE
        elif macro_bear and not medium_bear and trend_strong:
            # Macro bear but 4h bounce: wait for RSI extreme
            if rsi_extreme_short:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0 and not stoploss_triggered:
            # Exit long if macro trend reverses
            if not macro_bull and rsi_4h[i] > 50.0:
                desired_signal = 0.0
            # Exit long if RSI becomes overbought
            elif rsi_4h[i] > 70.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0 and not stoploss_triggered:
            # Exit short if macro trend reverses
            if not macro_bear and rsi_4h[i] < 50.0:
                desired_signal = 0.0
            # Exit short if RSI becomes oversold
            elif rsi_4h[i] < 30.0:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish
                if macro_bull and rsi_4h[i] < 65.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish
                if macro_bear and rsi_4h[i] > 35.0:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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