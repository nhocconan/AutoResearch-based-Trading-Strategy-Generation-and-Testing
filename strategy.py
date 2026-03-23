#!/usr/bin/env python3
"""
Experiment #881: 4h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: After 600+ failed strategies, a proven 4h HMA crossover system with
HTF trend alignment and regime-adaptive logic should work across ALL symbols.

Key insights from research:
1. 4h Primary TF: Target 20-50 trades/year (proven timeframe)
2. HMA(16/48) crossover: Fast HMA crosses slow HMA for trend changes
3. 1d HMA(21): Medium-term trend bias (only trade in HTF trend direction)
4. 1w HMA(21): Macro regime filter (bull/bear market)
5. RSI(14) pullback: Enter on RSI 40-60 pullback in trend direction
6. Choppiness Index(14): Switch between trend-follow and mean-revert modes
7. ATR(14) trailing stop (2.5x): Risk management

Why this should work:
- HMA crossover proven on SOL (Sharpe +0.879 in research)
- HTF alignment reduces false signals in counter-trend moves
- RSI pullback entries catch continuations, not tops/bottoms
- Choppiness filter adapts to market regime
- Discrete signal sizes minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crossover_rsi_chop_regime_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
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
    CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    hma_fast_4h = calculate_hma(close, 16)
    hma_slow_4h = calculate_hma(close, 48)
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_50_4h = calculate_sma(close, 50)
    sma_200_4h = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime (bull/bear market)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    # Track HMA crossover state
    prev_hma_diff = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(hma_fast_4h[i]) or np.isnan(hma_slow_4h[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_50_4h[i]) or np.isnan(sma_200_4h[i]):
            continue
        
        # HMA crossover signal
        hma_diff = hma_fast_4h[i] - hma_slow_4h[i]
        prev_hma_diff_prev = hma_fast_4h[i-1] - hma_slow_4h[i-1] if i > 0 else 0.0
        
        hma_bullish_cross = prev_hma_diff_prev <= 0 and hma_diff > 0
        hma_bearish_cross = prev_hma_diff_prev >= 0 and hma_diff < 0
        hma_bullish = hma_diff > 0
        hma_bearish = hma_diff < 0
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SHORT-TERM TREND FILTER (4h SMA50/200) ===
        above_sma50 = close[i] > sma_50_4h[i]
        below_sma50 = close[i] < sma_50_4h[i]
        above_sma200 = close[i] > sma_200_4h[i]
        below_sma200 = close[i] < sma_200_4h[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === RSI SIGNALS ===
        rsi_neutral_long = 40 <= rsi_4h[i] <= 60
        rsi_neutral_short = 40 <= rsi_4h[i] <= 60
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 45) — HMA Crossover + RSI Pullback ===
        if trending_regime:
            # Long: HMA bullish + HTF alignment + RSI pullback
            if hma_bullish and (macro_bull or trend_1d_bullish) and rsi_neutral_long:
                desired_signal = BASE_SIZE
            # Strong long: HMA cross + all trend alignment
            elif hma_bullish_cross and macro_bull and trend_1d_bullish and above_sma50:
                desired_signal = BASE_SIZE
            # Pullback long in uptrend
            elif hma_bullish and macro_bull and rsi_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: HMA bearish + HTF alignment + RSI pullback
            if hma_bearish and (macro_bear or trend_1d_bearish) and rsi_neutral_short:
                desired_signal = -BASE_SIZE
            # Strong short: HMA cross + all trend alignment
            elif hma_bearish_cross and macro_bear and trend_1d_bearish and below_sma50:
                desired_signal = -BASE_SIZE
            # Pullback short in downtrend
            elif hma_bearish and macro_bear and rsi_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        elif ranging_regime:
            # Long: RSI oversold + price above long-term support
            if rsi_extreme_oversold and above_sma200:
                desired_signal = REDUCED_SIZE
            elif rsi_oversold and (macro_bull or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            # Short: RSI overbought + price below long-term resistance
            if rsi_extreme_overbought and below_sma200:
                desired_signal = -REDUCED_SIZE
            elif rsi_overbought and (macro_bear or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: HMA direction + HTF confirmation
            if hma_bullish and macro_bull and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            if hma_bearish and macro_bear and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Fallback: RSI extremes with SMA filter
            if rsi_extreme_oversold and above_sma200 and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            if rsi_extreme_overbought and below_sma200 and desired_signal == 0:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish and RSI not overbought
                if hma_bullish and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HMA still bearish and RSI not oversold
                if hma_bearish and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HMA turns bearish + RSI overbought
            if hma_bearish and rsi_4h[i] > 70:
                desired_signal = 0.0
            # Exit if macro trend reverses strongly
            if macro_bear and trend_1d_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HMA turns bullish + RSI oversold
            if hma_bullish and rsi_4h[i] < 30:
                desired_signal = 0.0
            # Exit if macro trend reverses strongly
            if macro_bull and trend_1d_bullish:
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