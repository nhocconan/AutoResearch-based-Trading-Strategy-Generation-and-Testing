#!/usr/bin/env python3
"""
Experiment #973: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Daily timeframe with weekly trend filter should produce 25-50 trades/year
with better risk-adjusted returns than lower timeframes. After 700+ failed experiments,
the key is LOOSER entry conditions to ensure trades happen on ALL symbols.

Key components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 15 (relaxed from 10 to ensure trades)
   - Short when CRSI > 85 (relaxed from 90)
2. Choppiness Index (CHOP): Regime filter
   - CHOP > 55 = range (mean reversion OK)
   - CHOP < 45 = trend (trend following)
3. 1w HMA(21): Macro trend bias (not absolute filter - just adds confluence)
4. ATR(14) trailing stop: 2.5x ATR

Critical fix from failed experiments:
- RELAXED CRSI thresholds (15/85 not 10/90) to guarantee trades
- Weekly HMA is confluence NOT absolute filter (allows trades in both directions)
- Simpler logic = fewer conflicting conditions = more trades
- Conservative sizing (0.25-0.30) controls drawdown

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_atr_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with ~75% win rate.
    RELAXED thresholds for more trades.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        pos_streak = np.sum(streak_window > 0)
        neg_streak = np.sum(streak_window < 0)
        total = pos_streak + neg_streak
        if total > 0:
            streak_rsi[i] = 100 * pos_streak / total
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current return stands vs lookback
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns) * 100
            percent_rank[i] = rank
    
    # Combine all three components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending.
    > 55 = ranging, < 45 = trending.
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA(21) for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    
    # Simple RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period=100 + warmup
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA21) - Confluence not filter ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        ranging_regime = chop_1d[i] > 55
        trending_regime = chop_1d[i] < 45
        
        # === CRSI SIGNALS - RELAXED for more trades ===
        crsi_extreme_low = crsi_1d[i] < 15  # Was 10
        crsi_extreme_high = crsi_1d[i] > 85  # Was 90
        crsi_oversold = crsi_1d[i] < 30
        crsi_overbought = crsi_1d[i] > 70
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        desired_signal = 0.0
        
        # === RANGING REGIME - Mean Reversion (primary strategy) ===
        if ranging_regime:
            # Long: CRSI extreme low (strong signal)
            if crsi_extreme_low:
                desired_signal = BASE_SIZE
            # Long: CRSI oversold + RSI oversold (confluence)
            elif crsi_oversold and rsi_oversold:
                desired_signal = REDUCED_SIZE
            # Long: CRSI low + macro bull support
            elif crsi_1d[i] < 25 and macro_bull:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI extreme high
            if crsi_extreme_high:
                desired_signal = -BASE_SIZE
            # Short: CRSI overbought + RSI overbought
            elif crsi_overbought and rsi_overbought:
                desired_signal = -BASE_SIZE
            # Short: CRSI high + macro bear support
            elif crsi_1d[i] > 75 and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME - Trend Following ===
        elif trending_regime:
            # Long: Macro bull + CRSI pullback (buy dip in uptrend)
            if macro_bull and crsi_oversold:
                desired_signal = BASE_SIZE
            # Long: Macro bull + CRSI low
            elif macro_bull and crsi_1d[i] < 35:
                desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + CRSI rally (sell rip in downtrend)
            if macro_bear and crsi_overbought:
                desired_signal = -BASE_SIZE
            # Short: Macro bear + CRSI high
            elif macro_bear and crsi_1d[i] > 65:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extremes only
            if crsi_extreme_low:
                desired_signal = BASE_SIZE
            elif crsi_1d[i] < 20 and macro_bull:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_high:
                desired_signal = -BASE_SIZE
            elif crsi_1d[i] > 80 and macro_bear:
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
        
        # === HOLD LOGIC - Maintain position through minor pullbacks ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought
                if crsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold
                if crsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI very overbought
            if crsi_1d[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI very oversold
            if crsi_1d[i] < 15:
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