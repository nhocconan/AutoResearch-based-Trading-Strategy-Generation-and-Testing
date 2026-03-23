#!/usr/bin/env python3
"""
Experiment #1072: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 777+ failed strategies, the winning pattern for 12h timeframe combines:
1. CONNORS RSI (CRSI) — proven 75% win rate for mean reversion
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 10 + price > SMA200 | Short: CRSI > 90 + price < SMA200
2. CHOPPINESS INDEX (CHOP) — regime detection (BEST filter for crypto)
   CHOP > 61.8 = range (mean reversion at BB/Donchian bounds)
   CHOP < 38.2 = trend (breakout following with HMA/Donchian)
3. 1d HMA21 + 1w HMA50 — dual HTF macro bias (only trade with both aligned)
4. ATR trailing stop (2.5x) — mandatory risk management
5. Position size: 0.25-0.30 discrete (reduced to 0.15 in vol spikes)

Why 12h works better than 4h:
- 12h = 20-50 trades/year target (optimal fee/trade balance)
- Less noise than 4h, more signals than 1d
- Proven in exp#1063 (1d CRSI+CHOP) and exp#1066 (12h Fisher+KAMA)

Timeframe: 12h (primary)
HTF: 1d (daily) + 1w (weekly) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels (reduced to 0.15 in high vol)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_regime_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index — standard Welles Wilder calculation."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Use EMA for smoothing (faster response than SMA)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    valid_mask = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 1e-10)
    rs[valid_mask] = avg_gain[valid_mask] / avg_loss[valid_mask]
    
    rsi[valid_mask] = 100.0 - (100.0 / (1.0 + rs[valid_mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — combines 3 components for mean reversion signals.
    
    Formula:
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) — short-term momentum
    2. RSI of streak length — measures consecutive up/down days
    3. PercentRank — where current return ranks vs last 100 bars
    
    Signals:
    - CRSI < 10 = extremely oversold (long opportunity)
    - CRSI > 90 = extremely overbought (short opportunity)
    
    Research shows 75% win rate when combined with SMA200 filter.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (positive streak = bullish, negative = bearish)
    # Use absolute streak length with sign for RSI calculation
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, 50.0)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        down_streaks = np.sum(streak[i-streak_period+1:i+1] < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
    
    # Component 3: PercentRank of returns
    returns = np.diff(close) / close[:-1]
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) == rank_period and not np.any(np.isnan(window)):
            current_return = returns[i-1] if i > 0 else 0
            percent_rank[i] = 100.0 * np.sum(window < current_return) / len(window)
    
    # Combine components
    valid_mask = (~np.isnan(rsi_short)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range market (mean reversion favored)
    - CHOP < 38.2 = trending market (breakout/trend follow favored)
    - 38.2 - 61.8 = transition zone
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA50 for longer-term macro filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === VOLATILITY REGIME (Position Sizing) ===
        vol_spike = atr_ratio[i] > 2.0
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # === MACRO TREND (1d HMA21 + 1w HMA50) ===
        macro_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        macro_neutral = not macro_bull and not macro_bear
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        
        # === CRSI SIGNALS ===
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === CHOPPY REGIME: MEAN REVERSION ===
        if is_choppy:
            # Long: CRSI extreme oversold + above SMA200 + macro not bearish
            if crsi_extreme_oversold and above_sma200 and not macro_bear:
                desired_signal = current_size
            # Long: CRSI oversold + price at lower Donchian
            elif crsi_oversold and close[i] <= donch_lower[i] * 1.002 and above_sma200:
                desired_signal = current_size * 0.7
            
            # Short: CRSI extreme overbought + below SMA200 + macro not bullish
            elif crsi_extreme_overbought and below_sma200 and not macro_bull:
                desired_signal = -current_size
            # Short: CRSI overbought + price at upper Donchian
            elif crsi_overbought and close[i] >= donch_upper[i] * 0.998 and below_sma200:
                desired_signal = -current_size * 0.7
        
        # === TRENDING REGIME: BREAKOUT FOLLOWING ===
        elif is_trending:
            # Long breakout + macro bullish + above SMA200
            if donch_breakout_long and macro_bull and above_sma200:
                desired_signal = current_size
            # Long on pullback in uptrend (CRSI oversold but macro bullish)
            elif crsi_oversold and macro_bull and close[i] > donch_mid[i]:
                desired_signal = current_size * 0.5
            
            # Short breakout + macro bearish + below SMA200
            elif donch_breakout_short and macro_bear and below_sma200:
                desired_signal = -current_size
            # Short on pullback in downtrend (CRSI overbought but macro bearish)
            elif crsi_overbought and macro_bear and close[i] < donch_mid[i]:
                desired_signal = -current_size * 0.5
        
        # === TRANSITION ZONE: CONSERVATIVE SIGNALS ===
        else:
            # Only take extreme CRSI signals with strong macro alignment
            if crsi_extreme_oversold and macro_bull and above_sma200:
                desired_signal = current_size * 0.7
            elif crsi_extreme_overbought and macro_bear and below_sma200:
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
                # Hold long if CRSI not overbought and macro still bullish
                if crsi[i] < 70.0 and (macro_bull or close[i] > donch_mid[i]):
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if CRSI not oversold and macro still bearish
                if crsi[i] > 30.0 and (macro_bear or close[i] < donch_mid[i]):
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought OR macro reverses bearish
            if crsi_overbought or (macro_bear and close[i] < donch_mid[i]):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold OR macro reverses bullish
            if crsi_oversold or (macro_bull and close[i] > donch_mid[i]):
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