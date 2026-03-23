#!/usr/bin/env python3
"""
Experiment #967: 1d Primary + 1w HTF — Connors RSI + HMA Trend + Choppiness Regime

Hypothesis: Daily timeframe with weekly trend filter should capture major moves while
avoiding whipsaw. Connors RSI (CRSI) is proven mean-reversion indicator with 75% win rate.
Combined with 1w HMA for macro trend and Choppiness Index for regime detection.

Why 1d timeframe:
- Target 20-50 trades/year (minimal fee drag ~1-2.5%)
- Higher timeframes work best for BTC/ETH (proven in research)
- Less noise than 4h/1h, cleaner signals
- Works through 2022 crash and 2025 bear market

Key components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > 1w HMA(21)
   - Short: CRSI > 85 + price < 1w HMA(21)
2. Choppiness Index regime filter: CHOP > 55 = range (use CRSI), CHOP < 45 = trend (use breakout)
3. 1w HMA(21) for macro trend bias (aligned properly via mtf_data)
4. ATR(14) trailing stoploss at 3x ATR
5. Donchian(20) breakout for trending regime entries

Position sizing: 0.25-0.30 discrete levels to minimize fee churn
Stoploss: 3x ATR trailing stop via signal → 0

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_hma_chop_regime_1w_atr_v1"
timeframe = "1d"
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
    """RSI Streak component of Connors RSI.
    Measures consecutive up/down days as percentage."""
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return streak_rsi
    
    # Calculate streak values (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert to RSI-like score (0-100)
    for i in range(period, n):
        window = streak[i-period+1:i+1]
        up_sum = np.sum(np.where(window > 0, window, 0))
        down_sum = np.abs(np.sum(np.where(window < 0, window, 0)))
        
        if up_sum + down_sum > 0:
            streak_rsi[i] = 100 * up_sum / (up_sum + down_sum)
        else:
            streak_rsi[i] = 50
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of Connors RSI.
    Current close percentile vs last N days."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period:
        return pr
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / (period - 1) * 100
        pr[i] = rank
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period:
        return crsi
    
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    for i in range(pr_period - 1, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pr[i]) / 3
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel — highest high and lowest low over period."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, lower
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d SMA200 for additional trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
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
    
    for i in range(250, n):  # Start after enough data for all indicators
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        # === MACRO TREND (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === LONG-TERM TREND (SMA200) ===
        long_bull = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        long_bear = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop[i] > 55
        trending_regime = chop[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        donch_breakout_short = close[i] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI oversold + macro trend support
            if crsi_oversold and macro_bull:
                desired_signal = BASE_SIZE
            # Long: CRSI extreme oversold (stronger signal)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            # Long: CRSI oversold + SMA200 support
            elif crsi_oversold and long_bull:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + macro trend support
            if crsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE
            # Short: CRSI extreme overbought
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            # Short: CRSI overbought + SMA200 resistance
            elif crsi_overbought and long_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with Donchian ===
        elif trending_regime:
            # Long: Donchian breakout + macro bull
            if donch_breakout_long and macro_bull:
                desired_signal = BASE_SIZE
            # Long: Donchian breakout + long bull
            elif donch_breakout_long and long_bull:
                desired_signal = REDUCED_SIZE
            
            # Short: Donchian breakdown + macro bear
            if donch_breakout_short and macro_bear:
                desired_signal = -BASE_SIZE
            # Short: Donchian breakdown + long bear
            elif donch_breakout_short and long_bear:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extremes only with trend confluence
            if crsi_extreme_oversold and (macro_bull or long_bull):
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought and (macro_bear or long_bear):
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro trend still bull and CRSI not overbought
                if macro_bull and crsi[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro trend still bear and CRSI not oversold
                if macro_bear and crsi[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro trend reverses
            if macro_bear and crsi[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro trend reverses
            if macro_bull and crsi[i] < 30:
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
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
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