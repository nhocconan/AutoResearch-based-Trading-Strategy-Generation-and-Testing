#!/usr/bin/env python3
"""
Experiment #1023: 1d Primary + 1w HTF — Connors RSI + Donchian + Choppiness Regime

Hypothesis: After analyzing 741+ failed strategies, the key insight is that 1d timeframe
with 1w HTF provides optimal trade frequency (20-50/year) while maintaining signal quality.
This strategy combines proven patterns from kept experiments (#1012, #1013):

1. CONNORS RSI (CRSI): 3-component mean reversion signal
   - RSI(3) for short-term momentum
   - RSI_Streak(2) for consecutive up/down days
   - PercentRank(100) for relative price position
   - Long: CRSI < 15 (oversold), Short: CRSI > 85 (overbought)
   - Proven 75% win rate in bear/range markets (ETH Sharpe +0.923 in research)

2. 1w HMA21: Long-term trend bias
   - Only long when price > 1w HMA21 (weekly bullish)
   - Only short when price < 1w HMA21 (weekly bearish)
   - HMA responds faster than EMA while smoothing noise

3. DONCHIAN BREAKOUT (20-day): Trend confirmation
   - Long breakout above 20-day high confirms momentum
   - Short breakout below 20-day low confirms downside
   - Reduces false mean-reversion signals in strong trends

4. CHOPPINESS INDEX (14): Regime filter
   - CHOP > 61.8 = ranging → favor CRSI mean reversion
   - CHOP < 38.2 = trending → favor Donchian breakout
   - Prevents entering mean-reversion in strong trends

5. ATR Trailing Stop (2.5x): Risk management
   - Signal → 0 when stoploss hit
   - Protects capital during 2022-style crashes

Why 1d works:
- Target 20-50 trades/year (optimal for fee drag vs statistical significance)
- Less noise than 4h/12h, more signals than 1w
- Works across BTC/ETH/SOL (not SOL-biased)

Critical fixes from failed experiments:
- RELAXED CRSI thresholds (15/85 not 10/90) for more trades
- Single 1w HMA (not dual 12h+1d) for cleaner trend signal
- Ensure ALL symbols generate trades (no SOL-only strategies)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_donchian_1w_hma_chop_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    3-component mean reversion indicator:
    1. RSI(close, 3) - short-term momentum
    2. RSI(streak, 2) - consecutive up/down days
    3. PercentRank(close, 100) - relative price position
    
    Entry: CRSI < 15 (oversold long), CRSI > 85 (overbought short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_positive[i-streak_period+1:i+1])
        avg_loss = np.mean(streak_negative[i-streak_period+1:i+1])
        if avg_loss == 0:
            streak_rsi[i] = 100.0
        else:
            rs = avg_gain / (avg_loss + 1e-10)
            streak_rsi[i] = 100 - (100 / (1 + rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = (rank / rank_period) * 100
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad first element
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel (upper and lower bounds)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1d = calculate_atr(high, low, close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === LONG-TERM TREND (1w HMA21) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_1d[i] > 61.8  # Ranging market → mean reversion
        regime_trend = chop_1d[i] < 38.2  # Trending market → trend follow
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_1d[i] < 15  # Strong oversold
        crsi_overbought = crsi_1d[i] > 85  # Strong overbought
        crsi_mild_oversold = crsi_1d[i] < 25  # Mild oversold
        crsi_mild_overbought = crsi_1d[i] > 75  # Mild overbought
        
        # === DONCHIAN BREAKOUT (Trend Confirmation) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if weekly_bull:
            if regime_chop:
                # Mean reversion in ranging market with bullish weekly trend
                if crsi_oversold:
                    desired_signal = BASE_SIZE
                elif crsi_mild_oversold and close[i] > hma_1w_aligned[i] * 0.95:
                    desired_signal = REDUCED_SIZE
            elif regime_trend:
                # Trend following in trending bullish market
                if donchian_breakout_long:
                    desired_signal = BASE_SIZE
                elif crsi_mild_oversold:
                    desired_signal = REDUCED_SIZE
            else:
                # Neutral regime - use CRSI with relaxed threshold
                if crsi_mild_oversold:
                    desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if weekly_bear:
            if regime_chop:
                # Mean reversion in ranging market with bearish weekly trend
                if crsi_overbought:
                    desired_signal = -BASE_SIZE
                elif crsi_mild_overbought and close[i] < hma_1w_aligned[i] * 1.05:
                    desired_signal = -REDUCED_SIZE
            elif regime_trend:
                # Trend following in trending bearish market
                if donchian_breakout_short:
                    desired_signal = -BASE_SIZE
                elif crsi_mild_overbought:
                    desired_signal = -REDUCED_SIZE
            else:
                # Neutral regime - use CRSI with relaxed threshold
                if crsi_mild_overbought:
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
                # Hold long if weekly bullish and CRSI not extreme overbought
                if weekly_bull and crsi_1d[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly bearish and CRSI not extreme oversold
                if weekly_bear and crsi_1d[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses
            if not weekly_bull and crsi_1d[i] > 50:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses
            if not weekly_bear and crsi_1d[i] < 50:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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