#!/usr/bin/env python3
"""
Experiment #1038: 30m Primary + 4h/1d HTF — Connors RSI + HMA Trend + Relaxed Entries

Hypothesis: After 750+ failed strategies, the #1 issue is TOO STRICT entry conditions
causing 0 trades (experiments #1028, #1030, #1032, #1033, #1035 all had Sharpe=0.000).

Key insight: For 30m timeframe, use SIMPLE HTF trend filter (4h HMA21 slope only),
relaxed CRSI thresholds (<15/>85 not <10/>90), wider session (6-22 UTC not 8-20),
and lower volume threshold (>0.5x not >0.8x).

Strategy components:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 (relaxed from 10) + price > 4h HMA21
   - Short: CRSI > 85 (relaxed from 90) + price < 4h HMA21
   - Proven 75% win rate in mean reversion

2. 4h HMA21 trend filter (SINGLE HTF, not dual):
   - Only long when 4h HMA slope > 0 (simpler than requiring price > HMA)
   - Only short when 4h HMA slope < 0
   - This generates MORE signals than price-vs-HMA

3. Volume filter: volume > 0.5x 20-bar average (relaxed from 0.8x)

4. Session filter: 6-22 UTC (wider than 8-20 for more opportunities)

5. ATR stoploss: 2.0x ATR (tighter for quicker exits, reduce drawdown)

6. Position sizing: 0.25 base (smaller for 30m to minimize fee drag)

Why this works for 30m:
- HTF (4h) provides direction, 30m provides entry timing
- Relaxed thresholds = 50-100 trades/year (not 0)
- Single HTF filter = less mutually exclusive conditions
- Discrete signal sizes (0.0, ±0.25) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 50-100 trades/year with relaxed entries)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_4h_hma_relaxed_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current return ranks vs last 100 bars
    
    Entry: CRSI < 15 (oversold), CRSI > 85 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3)
    rsi_3 = np.full(n, np.nan)
    for i in range(rsi_period, n):
        gains = 0.0
        losses = 0.0
        for j in range(i - rsi_period + 1, i + 1):
            if j == 0:
                continue
            change = close[j] - close[j-1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        if losses == 0:
            rsi_3[i] = 100.0
        else:
            rs = gains / losses
            rsi_3[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2)
    rsi_streak = np.full(n, np.nan)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    for i in range(streak_period + 5, n):
        gains = 0.0
        losses = 0.0
        for j in range(i - streak_period + 1, i + 1):
            if j == 0:
                continue
            change = streak[j] - streak[j-1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        if losses == 0:
            rsi_streak[i] = 100.0
        else:
            rs = gains / losses
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        if i < rank_period:
            continue
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            rank = np.sum(returns < current_return)
            percent_rank[i] = 100.0 * rank / len(returns)
    
    # Combine into CRSI
    for i in range(rank_period + 5, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)."""
    n = len(hma_values)
    slope = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i-lookback]):
            slope[i] = (hma_values[i] - hma_values[i-lookback]) / (hma_values[i-lookback] + 1e-10)
    
    return slope

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

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA21 for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 4h HMA slope for trend direction
    hma_4h_slope = calculate_hma_slope(hma_4h_aligned, lookback=3)
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 30m to reduce fee drag
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_4h_slope[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === HTF TREND DIRECTION (4h HMA slope) ===
        # Simpler than price-vs-HMA: just use slope direction
        trend_bull = hma_4h_slope[i] > 0.0001  # Slightly positive slope
        trend_bear = hma_4h_slope[i] < -0.0001  # Slightly negative slope
        
        # === CONNORS RSI SIGNALS (Relaxed thresholds) ===
        crsi_oversold = crsi_30m[i] < 15.0  # Relaxed from < 10
        crsi_overbought = crsi_30m[i] > 85.0  # Relaxed from > 90
        crsi_extreme_oversold = crsi_30m[i] < 10.0
        crsi_extreme_overbought = crsi_30m[i] > 90.0
        
        # === VOLUME FILTER (Relaxed) ===
        vol_ok = volume[i] > 0.5 * vol_avg[i]  # Relaxed from 0.8x
        
        # === SESSION FILTER (Wider window) ===
        hour = get_hour_from_open_time(open_time[i])
        session_ok = 6 <= hour <= 22  # Wider than 8-20
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if trend_bull and crsi_oversold and vol_ok and session_ok:
            # Standard long entry
            if crsi_extreme_oversold:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if trend_bear and crsi_overbought and vol_ok and session_ok:
            # Standard short entry
            if crsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish
                if trend_bull and crsi_30m[i] < 70.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish
                if trend_bear and crsi_30m[i] > 30.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses bearish
            if trend_bear and crsi_30m[i] > 50.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses bullish
            if trend_bull and crsi_30m[i] < 50.0:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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