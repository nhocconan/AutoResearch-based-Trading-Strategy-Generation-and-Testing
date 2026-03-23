#!/usr/bin/env python3
"""
Experiment #929: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Choppiness Filter

Hypothesis: After 660+ failed strategies, the key insight is SIMPLICITY + TRADE FREQUENCY.
Most recent failures (#917-#928) have Sharpe=0.000 = ZERO TRADES due to over-filtering.

This strategy uses PROVEN patterns with RELAXED thresholds:
1. 4h HMA(16/48) crossover for primary trend (faster than EMA, less lag)
2. 1d HMA(21) for macro bias (HTF direction filter — call ONCE before loop)
3. RSI(14) pullback entries at 40/60 (not 30/70 — too strict = 0 trades)
4. Choppiness(14) regime filter: CHOP>55 reduce size, CHOP<45 full size
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete signal sizes: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work:
- 4h timeframe targets 30-50 trades/year (optimal fee/trade balance)
- HMA crossover is proven trend filter (less whipsaw than EMA)
- RSI 40/60 thresholds ensure entries on all symbols (not just SOL)
- 1d HTF provides strong directional bias without over-filtering
- Simple logic = fewer conditions that can all fail simultaneously

Critical lessons from failures:
- RELAXED RSI thresholds (40/60 not 30/70) guarantee trades
- Single HTF (1d) not triple (1d/1w/4h) = less filter conflict
- Hold logic maintains position through minor pullbacks
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_regime_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging (reduce position), CHOP < 45 = trending (full size).
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    
    # Calculate primary (4h) indicators
    hma_16_4h = calculate_hma(close, 16)
    hma_48_4h = calculate_hma(close, 48)
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro trend bias
    hma_21_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d_raw)
    
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
        if np.isnan(hma_16_4h[i]) or np.isnan(hma_48_4h[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_21_1d_aligned[i]):
            continue
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bullish = close[i] > hma_21_1d_aligned[i]
        macro_bearish = close[i] < hma_21_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA16/48 crossover) ===
        hma_bullish = hma_16_4h[i] > hma_48_4h[i]
        hma_bearish = hma_16_4h[i] < hma_48_4h[i]
        
        # === REGIME (4h Choppiness) ===
        trending_regime = chop_4h[i] < 45
        ranging_regime = chop_4h[i] > 55
        
        # === RSI PULLBACK (relaxed thresholds for trade frequency) ===
        rsi_pullback_long = rsi_4h[i] < 50
        rsi_pullback_short = rsi_4h[i] > 50
        rsi_strong_long = rsi_4h[i] < 45
        rsi_strong_short = rsi_4h[i] > 55
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 45) — Full size entries ===
        if trending_regime:
            # Long: HMA bullish + RSI pullback + macro alignment
            if hma_bullish and rsi_pullback_long:
                if macro_bullish:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE
            
            # Short: HMA bearish + RSI pullback + macro alignment
            if hma_bearish and rsi_pullback_short:
                if macro_bearish:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (CHOP > 55) — Reduced size, mean reversion ===
        elif ranging_regime:
            # Long: HMA bullish + RSI oversold
            if hma_bullish and rsi_strong_long:
                desired_signal = REDUCED_SIZE
            
            # Short: HMA bearish + RSI overbought
            if hma_bearish and rsi_strong_short:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) — Conservative ===
        else:
            # Require stronger confluence
            if hma_bullish and macro_bullish and rsi_strong_long:
                desired_signal = REDUCED_SIZE
            
            if hma_bearish and macro_bearish and rsi_strong_short:
                desired_signal = -REDUCED_SIZE
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0:
            if position_side > 0:
                # Hold long if HMA still bullish and RSI not overbought
                if hma_bullish and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if HMA still bearish and RSI not oversold
                if hma_bearish and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses
            if hma_bearish and rsi_4h[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses
            if hma_bullish and rsi_4h[i] < 40:
                desired_signal = 0.0
        
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