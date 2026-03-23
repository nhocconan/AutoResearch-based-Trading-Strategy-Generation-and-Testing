#!/usr/bin/env python3
"""
Experiment #1073: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + Weekly HMA

Hypothesis: After 777+ failed experiments, the winning pattern for 1d timeframe combines:
1. CONNORS RSI (CRSI) — proven 75% win rate in crypto research
   Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 15 (oversold) | Short: CRSI > 85 (overbought)
   Much more responsive than standard RSI(14) for daily entries
2. CHOPPINESS INDEX (CHOP) — regime detection
   CHOP > 61.8 = range (mean reversion at CRSI extremes)
   CHOP < 38.2 = trend (momentum continuation on CRSI crossover)
3. 1w HMA21 macro bias — only trade in direction of weekly trend
   Filters out counter-trend trades that failed in 2022 crash
4. ATR trailing stop (2.5x) — protects against 2022-style crashes
5. Discrete position sizing (0.28/0.15) — minimizes fee churn

Why this should beat Sharpe=0.612:
- CRSI is PROVEN for crypto mean reversion (different from failed RSI/STC strategies)
- 1d timeframe = 10-30 trades/year (optimal for fee/trade balance)
- Weekly HMA filter prevents disaster trades in bear markets
- Simpler logic = fewer 0-trade failures (major issue in experiments 1068-1072)
- Works on BTC/ETH/SOL individually (not SOL-biased)

Timeframe: 1d (primary)
HTF: 1w (weekly) — loaded ONCE before loop using mtf_data helper
Position Size: 0.28 base, 0.15 reduced (high vol)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, percentile_rank_period=100):
    """
    Connors RSI (CRSI) — combines 3 momentum components for mean reversion signals.
    
    Formula: CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) on close — short-term momentum
    2. RSI(2) on streak — direction persistence (consecutive up/down days)
    3. PercentRank(100) — where current close ranks vs last 100 days
    
    Signals:
    - CRSI < 10-15 = oversold (long opportunity)
    - CRSI > 85-90 = overbought (short opportunity)
    
    Research shows 75% win rate on daily crypto data with SMA200 filter.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < percentile_rank_period:
        return crsi
    
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on close
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close_vals = rsi_close.values
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak_vals = rsi_streak.values
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(percentile_rank_period, n):
        window = close[i - percentile_rank_period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (percentile_rank_period - 1)
    
    # Combine components
    valid_mask = (~np.isnan(rsi_close_vals)) & (~np.isnan(rsi_streak_vals)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_close_vals[valid_mask] + rsi_streak_vals[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range market (mean reversion favored)
    - CHOP < 38.2 = trending market (breakout/trend follow favored)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR Ratio for volatility regime detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan)
    valid_mask = (~np.isnan(atr_short)) & (~np.isnan(atr_long)) & (atr_long > 1e-10)
    ratio[valid_mask] = atr_short[valid_mask] / atr_long[valid_mask]
    
    return ratio

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, percentile_rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track previous values for crossover detection
    prev_crsi = np.full(n, 50.0)
    prev_chop = np.full(n, 50.0)
    for i in range(1, n):
        if not np.isnan(crsi[i-1]):
            prev_crsi[i] = crsi[i-1]
        if not np.isnan(chop[i-1]):
            prev_chop[i] = chop[i-1]
    
    for i in range(250, n):  # Warmup for SMA200 + CRSI
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        
        # === VOLATILITY REGIME (Position Sizing) ===
        vol_spike = atr_ratio[i] > 2.0
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # === MACRO TREND FILTERS ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        sma_bull = close[i] > sma_200[i]
        sma_bear = close[i] < sma_200[i]
        
        # Strong macro signals (both weekly HMA and SMA200 agree)
        macro_bull = weekly_bull and sma_bull
        macro_bear = weekly_bear and sma_bear
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # CRSI crossover signals (more reliable than absolute levels)
        crsi_long_cross = prev_crsi[i] < 15.0 and crsi[i] >= 15.0
        crsi_short_cross = prev_crsi[i] > 85.0 and crsi[i] <= 85.0
        
        # Extreme CRSI for mean reversion
        crsi_extreme_long = crsi[i] < 10.0
        crsi_extreme_short = crsi[i] > 90.0
        
        desired_signal = 0.0
        
        # === CHOPPY REGIME: MEAN REVERSION AT CRSI EXTREMES ===
        if is_choppy:
            # Long at extreme oversold + macro not strongly bearish
            if crsi_extreme_long and not macro_bear:
                desired_signal = current_size
            elif crsi_oversold and (weekly_bull or not weekly_bear):
                desired_signal = current_size * 0.5
            
            # Short at extreme overbought + macro not strongly bullish
            elif crsi_extreme_short and not macro_bull:
                desired_signal = -current_size
            elif crsi_overbought and (weekly_bear or not weekly_bull):
                desired_signal = -current_size * 0.5
        
        # === TRENDING REGIME: MOMENTUM CONTINUATION ===
        elif is_trending:
            # Long on CRSI crossover + macro bullish
            if crsi_long_cross and macro_bull:
                desired_signal = current_size
            elif crsi_oversold and macro_bull:
                desired_signal = current_size * 0.5
            
            # Short on CRSI crossover + macro bearish
            elif crsi_short_cross and macro_bear:
                desired_signal = -current_size
            elif crsi_overbought and macro_bear:
                desired_signal = -current_size * 0.5
        
        # === TRANSITION ZONE: COMBINED SIGNALS ===
        else:
            # Long: CRSI oversold + weekly trend not bearish
            if crsi_oversold and not weekly_bear:
                desired_signal = current_size * 0.5
            elif crsi_long_cross and weekly_bull:
                desired_signal = current_size
            
            # Short: CRSI overbought + weekly trend not bullish
            if crsi_overbought and not weekly_bull:
                desired_signal = -current_size * 0.5
            elif crsi_short_cross and weekly_bear:
                desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and macro not bearish
                if crsi[i] < 80.0 and not macro_bear:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if CRSI not oversold and macro not bullish
                if crsi[i] > 20.0 and not macro_bull:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought OR macro reverses bearish
            if crsi_overbought:
                desired_signal = 0.0
            elif macro_bear and crsi[i] > 50.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold OR macro reverses bullish
            if crsi_oversold:
                desired_signal = 0.0
            elif macro_bull and crsi[i] < 50.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.7:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.7:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.7:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.7:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        
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
                # Flip position
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