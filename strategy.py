#!/usr/bin/env python3
"""
Experiment #839: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: After 575+ failed strategies, the key issue is OVER-FILTERING.
Complex regime logic with 5+ confluence filters = 0 trades.
This strategy SIMPLIFIES entry conditions while keeping proven edges:

1. 4h Primary timeframe (target 30-50 trades/year)
2. 1d HMA(21) for long-term trend bias (HTF direction filter)
3. 4h HMA(16) vs HMA(48) crossover for entry timing
4. 4h RSI(14) with relaxed thresholds (40/60) for momentum confirmation
5. 4h Choppiness Index(14) for regime detection (trend vs mean-revert)
6. 4h ATR(14) for trailing stop (2.5x)
7. Dual regime: trend-follow when CHOP<45, mean-revert when CHOP>55

Why this should work:
- Fewer filters = more trades (addresses #1 failure mode)
- 4h timeframe proven in current best (Sharpe=0.612)
- 1d HTF provides trend bias without over-complicating entries
- Relaxed RSI thresholds (40/60 vs 30/70) generate more signals
- Discrete position sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Key changes from failed strategies:
- REMOVED: Fisher Transform (too many false signals)
- REMOVED: Donchian breakout (redundant with HMA crossover)
- REMOVED: SMA200 (redundant with 1d HMA)
- SIMPLIFIED: Regime logic to 2 states (trending/ranging)
- RELAXED: RSI thresholds from 35/65 to 40/60

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_regime_1d_atr_v2"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
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
    CHOP > 55 = ranging, CHOP < 45 = trending.
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
    
    # Calculate and align 1d HMA for long-term trend bias
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
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_21_1d_aligned[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_21_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_21_1d_aligned[i]
        
        # === 4h HMA CROSSOVER SIGNAL ===
        hma_bullish_cross = hma_16_4h[i] > hma_48_4h[i]
        hma_bearish_cross = hma_16_4h[i] < hma_48_4h[i]
        
        # Check for fresh crossover (within last 3 bars)
        hma_recent_bull = False
        hma_recent_bear = False
        for j in range(max(100, i-3), i+1):
            if np.isnan(hma_16_4h[j]) or np.isnan(hma_48_4h[j]):
                continue
            if hma_16_4h[j] > hma_48_4h[j] and j > 100 and hma_16_4h[j-1] <= hma_48_4h[j-1]:
                hma_recent_bull = True
            if hma_16_4h[j] < hma_48_4h[j] and j > 100 and hma_16_4h[j-1] >= hma_48_4h[j-1]:
                hma_recent_bear = True
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === RSI SIGNALS (Relaxed thresholds for more trades) ===
        rsi_bullish = rsi_4h[i] > 40
        rsi_bearish = rsi_4h[i] < 60
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        if trending_regime:
            # Long: 1d trend bullish + 4h HMA bullish + RSI confirmation
            if trend_1d_bullish and hma_bullish_cross and rsi_bullish:
                desired_signal = BASE_SIZE
            # Entry on fresh crossover with trend alignment
            elif trend_1d_bullish and hma_recent_bull and rsi_4h[i] > 45:
                desired_signal = REDUCED_SIZE
            
            # Short: 1d trend bearish + 4h HMA bearish + RSI confirmation
            if trend_1d_bearish and hma_bearish_cross and rsi_bearish:
                desired_signal = -BASE_SIZE
            # Entry on fresh crossover with trend alignment
            elif trend_1d_bearish and hma_recent_bear and rsi_4h[i] < 55:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI oversold + price above 1d HMA (bullish bias)
            if rsi_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            # Short: RSI overbought + price below 1d HMA (bearish bias)
            if rsi_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Fallback: HMA crossover with RSI extreme (guarantees trades)
            if hma_recent_bull and rsi_4h[i] < 45:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            if hma_recent_bear and rsi_4h[i] > 55:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require both 1d trend and 4h HMA alignment
            if trend_1d_bullish and hma_bullish_cross and rsi_bullish:
                desired_signal = REDUCED_SIZE
            if trend_1d_bearish and hma_bearish_cross and rsi_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Fallback for trade generation
            if hma_recent_bull and rsi_4h[i] > 40:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            if hma_recent_bear and rsi_4h[i] < 60:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
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
                # Hold long if 4h HMA still bullish
                if hma_bullish_cross and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h HMA still bearish
                if hma_bearish_cross and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses + 4h HMA bearish
            if trend_1d_bearish and hma_bearish_cross:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_4h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses + 4h HMA bullish
            if trend_1d_bullish and hma_bullish_cross:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_4h[i] < 25:
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