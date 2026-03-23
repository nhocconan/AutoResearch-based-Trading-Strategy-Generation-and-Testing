#!/usr/bin/env python3
"""
Experiment #1082: 12h Primary + 1d/1w HTF — Dual Regime Strategy

Hypothesis: 12h timeframe with dual regime switching can achieve 20-50 trades/year
with better risk-adjusted returns than pure trend or pure mean-reversion.

Key Components:
1. CHOPPINESS INDEX (14) — regime detection
   CHOP > 55 = range (use mean reversion logic)
   CHOP < 45 = trend (use breakout logic)
   45-55 = transition (reduced size)
   
2. MEAN REVERSION MODE (choppy):
   - Connors RSI < 15 + price > 1d HMA50 → long
   - Connors RSI > 85 + price < 1d HMA50 → short
   - Exit at RSI 50/50 or Donchian mid
   
3. TREND MODE (trending):
   - Donchian(20) breakout + 1d HMA21 alignment → enter
   - Trail with 2.5x ATR
   
4. 1d HMA21/50 — macro trend bias (only trade with daily trend)
5. 1w HMA50 — weekly macro filter (avoid counter-weekly trades)
6. ATR(14) — volatility measurement and stoploss

Why this should work on 12h:
- Looser CRSI thresholds (15/85 vs 10/90) = more trades
- Dual regime = adapts to market conditions
- 12h bars = fewer false breakouts than 4h/1h
- Position size 0.25-0.30 with vol scaling

Timeframe: 12h (primary)
HTF: 1d, 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_chop_donchian_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — composite mean reversion indicator.
    
    Formula:
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) on close — short-term momentum
    2. RSI(2) on streak — streak duration (consecutive up/down days)
    3. PercentRank(100) — where current price ranks vs last 100 bars
    
    Signals:
    - CRSI < 10-15 = oversold (long opportunity)
    - CRSI > 85-90 = overbought (short opportunity)
    
    Proven 75% win rate in research for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + rsi_period + streak_period:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.zeros(n)
    valid = avg_loss > 1e-10
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rs[~valid] = 100.0  # No loss = RSI 100
    
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute for RSI calculation
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.zeros(n)
    streak_valid = streak_avg_loss > 1e-10
    streak_rs[streak_valid] = streak_avg_gain[streak_valid] / streak_avg_loss[streak_valid]
    streak_rs[~streak_valid] = 100.0
    
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if np.any(np.isnan(window)):
            continue
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    valid_mask = (~np.isnan(rsi_close)) & (~np.isnan(rsi_streak)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 55-61.8 = choppy/range market (mean reversion favored)
    - CHOP < 38.2-45 = trending market (breakout/trend follow favored)
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
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_21_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21_raw)
    
    hma_1d_50_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50_raw)
    
    # Calculate and align 1w HMA for weekly macro filter
    hma_1w_50_raw = calculate_hma(df_1w['close'].values, 50)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50_raw)
    
    # Calculate primary (12h) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    
    # 12h SMA for additional trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20  # Reduced in transition zone or high vol
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track CRSI for mean reversion entries
    prev_crsi = np.full(n, 50.0)
    for i in range(1, n):
        if not np.isnan(crsi[i-1]):
            prev_crsi[i] = crsi[i-1]
    
    for i in range(250, n):  # Need enough data for all indicators
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        # === VOLATILITY REGIME (Position Sizing) ===
        vol_spike = atr_ratio[i] > 2.0
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # === MACRO TREND FILTERS ===
        # Daily trend
        daily_bull = close[i] > hma_1d_21_aligned[i]
        daily_bear = close[i] < hma_1d_21_aligned[i]
        daily_strong_bull = close[i] > hma_1d_50_aligned[i]
        daily_strong_bear = close[i] < hma_1d_50_aligned[i]
        
        # Weekly trend (only if available)
        weekly_bull = True
        weekly_bear = True
        if not np.isnan(hma_1w_50_aligned[i]):
            weekly_bull = close[i] > hma_1w_50_aligned[i]
            weekly_bear = close[i] < hma_1w_50_aligned[i]
        
        # 12h trend
        trend_bull = close[i] > sma_50[i] and sma_50[i] > sma_200[i]
        trend_bear = close[i] < sma_50[i] and sma_50[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market (looser threshold for more trades)
        is_trending = chop[i] < 45.0  # Trend market
        is_transition = not is_choppy and not is_trending  # 45-55
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 20.0  # Looser threshold for more trades
        crsi_overbought = crsi[i] > 80.0  # Looser threshold
        
        crsi_cross_up = prev_crsi[i] < 20.0 and crsi[i] >= 20.0
        crsi_cross_down = prev_crsi[i] > 80.0 and crsi[i] <= 80.0
        
        # === DONCHIAN BREAKOUT (Trend Following) ===
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === CHOPPY REGIME: MEAN REVERSION ===
        if is_choppy:
            # Long: CRSI oversold + daily bullish bias + above weekly support
            if crsi_oversold and (daily_bull or daily_strong_bull):
                if weekly_bull or np.isnan(hma_1w_50_aligned[i]):
                    desired_signal = current_size
            # Short: CRSI overbought + daily bearish bias + below weekly resistance
            elif crsi_overbought and (daily_bear or daily_strong_bear):
                if weekly_bear or np.isnan(hma_1w_50_aligned[i]):
                    desired_signal = -current_size
        
        # === TRENDING REGIME: BREAKOUT FOLLOWING ===
        elif is_trending:
            # Long breakout + daily bullish + trend confirmation
            if donch_breakout_long and daily_bull:
                if trend_bull or close[i] > sma_50[i]:
                    desired_signal = current_size
            # Short breakout + daily bearish + trend confirmation
            elif donch_breakout_short and daily_bear:
                if trend_bear or close[i] < sma_50[i]:
                    desired_signal = -current_size
        
        # === TRANSITION ZONE: COMBINED SIGNALS (reduced size) ===
        elif is_transition:
            current_size = REDUCED_SIZE
            # Long: CRSI crossover + daily bullish
            if crsi_cross_up and daily_bull:
                desired_signal = current_size
            # Short: CRSI crossover + daily bearish
            elif crsi_cross_down and daily_bear:
                desired_signal = -current_size
            # Breakout with strong confirmation
            elif donch_breakout_long and daily_strong_bull:
                desired_signal = current_size * 0.7
            elif donch_breakout_short and daily_strong_bear:
                desired_signal = -current_size * 0.7
        
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
                # Hold long if CRSI not overbought or price above Donchian mid
                if crsi[i] < 70.0 or close[i] > donch_mid[i]:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if CRSI not oversold or price below Donchian mid
                if crsi[i] > 30.0 or close[i] < donch_mid[i]:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought in choppy market
            if is_choppy and crsi_overbought:
                desired_signal = 0.0
            # Exit long if daily trend reverses strongly
            if daily_strong_bear and crsi[i] > 50.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold in choppy market
            if is_choppy and crsi_oversold:
                desired_signal = 0.0
            # Exit short if daily trend reverses strongly
            if daily_strong_bull and crsi[i] < 50.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
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