#!/usr/bin/env python3
"""
Experiment #949: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 678 failed strategies, the key is SIMPLER entry conditions that 
actually fire while maintaining quality. Connors RSI (CRSI) has 75% win rate for 
mean reversion. Combined with Choppiness Index regime filter and 1d HMA trend bias,
this should generate 30-50 trades/year with positive Sharpe across ALL symbols.

Why this should work:
1. CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven mean reversion
2. Choppiness Index: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 1d HMA(21) for macro trend bias — only trade with macro direction in trend regime
4. RELAXED entry thresholds to ensure trades fire (CRSI < 15 not < 5)
5. ATR(14) trailing stop at 2.5x for risk management

Key improvements over #934:
- REMOVED funding rate dependency (causes 0 trades when data unavailable)
- SIMPLER entry logic (2-3 conditions max, not 5-6)
- CRSI instead of RSI (better for mean reversion, proven 75% win rate)
- Discrete signal sizes: 0.0, ±0.25, ±0.30 (minimize fee churn)
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_rsi_streak(close, period=2):
    """RSI Streak: consecutive up/down days."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return streak_rsi
    
    for i in range(period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, i - period, -1):
            if j > 0:
                if close[j] > close[j-1]:
                    up_streak += 1
                    down_streak = 0
                elif close[j] < close[j-1]:
                    down_streak += 1
                    up_streak = 0
                else:
                    break
        
        # Convert streak to RSI-like scale (0-100)
        if up_streak > 0:
            streak_rsi[i] = 100 * up_streak / period
        elif down_streak > 0:
            streak_rsi[i] = 100 * (1 - down_streak / period)
        else:
            streak_rsi[i] = 50
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank: where current price ranks in lookback period."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period:
        return pr
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        pr[i] = 100 * rank / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period:
        return crsi
    
    rsi_3 = calculate_rsi(close, rsi_period)
    streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    for i in range(pr_period - 1, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + streak[i] + pr[i]) / 3
    
    return crsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    sma_200_4h = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(chop_4h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200_4h[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_4h[i] < 15  # RELAXED from < 5 to ensure trades
        crsi_overbought = crsi_4h[i] > 85  # RELAXED from > 95
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200_4h[i]
        below_sma200 = close[i] < sma_200_4h[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + above SMA200 (bullish bias in range)
            if crsi_oversold and above_sma200:
                desired_signal = BASE_SIZE
            # Long: CRSI extreme oversold (stronger signal)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + below SMA200 (bearish bias in range)
            if crsi_overbought and below_sma200:
                desired_signal = -BASE_SIZE
            # Short: CRSI extreme overbought (stronger signal)
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Macro bull + CRSI pullback (buy dip in uptrend)
            if macro_bull and crsi_oversold:
                desired_signal = BASE_SIZE
            # Long: Macro bull + CRSI recovering from extreme
            elif macro_bull and crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + CRSI rally (sell rip in downtrend)
            if macro_bear and crsi_overbought:
                desired_signal = -BASE_SIZE
            # Short: Macro bear + CRSI extreme overbought
            elif macro_bear and crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only extreme CRSI signals
            if crsi_extreme_oversold and (macro_bull or above_sma200):
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought and (macro_bear or below_sma200):
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought yet
                if crsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold yet
                if crsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought
            if crsi_overbought:
                desired_signal = 0.0
            # Exit if macro reverses bearish
            if macro_bear and crsi_4h[i] > 50:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold
            if crsi_oversold:
                desired_signal = 0.0
            # Exit if macro reverses bullish
            if macro_bull and crsi_4h[i] < 50:
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