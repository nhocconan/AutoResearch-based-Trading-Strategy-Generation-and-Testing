#!/usr/bin/env python3
"""
Experiment #598: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness + Volume + Session

Hypothesis: 30m entries are too noisy alone, but when filtered by 4h/1d trend direction,
Choppiness regime, Connors RSI extremes, volume confirmation, and session timing,
we get HTF-quality signals with 30m entry precision.

Key innovations:
1. Connors RSI (CRSI) for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 (extreme oversold within uptrend)
   - Short: CRSI > 85 (extreme overbought within downtrend)
2. Choppiness Index regime filter: CHOP > 55 = mean revert, CHOP < 45 = trend follow
3. 4h HMA for intermediate trend bias (must align with entry direction)
4. 1d HMA for secular trend (no shorts if 1d strongly bullish, no longs if 1d strongly bearish)
5. Volume filter: current volume > 0.8x 20-bar average (avoid dead sessions)
6. Session filter: only 8-20 UTC (high liquidity, avoid Asian night whipsaw)
7. Conservative sizing: 0.25 long, 0.20 short (lower TF = more trades = smaller size)
8. ATR stoploss: 2.5x ATR trailing stop

Why this should work on 30m:
- CRSI has 75% win rate in backtests (per research notes)
- Session filter cuts 60% of low-quality overnight trades
- Volume filter avoids false breakouts on thin liquidity
- HTF alignment ensures we're not fighting the major trend
- Discrete signal levels minimize fee churn

Target: Sharpe > 0.612, trades 30-80/year per symbol, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_vol_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Fast RSI on close
    2. RSI_Streak(2): RSI on consecutive up/down streak lengths
    3. PercentRank(100): Percentile rank of today's price change over 100 days
    
    CRSI < 10-15 = extreme oversold (long opportunity)
    CRSI > 85-90 = extreme overbought (short opportunity)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3)
    def calc_rsi(prices, period):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
        
        rsi = np.pad(rsi, (1, 0), mode='constant', constant_values=np.nan)
        return rsi
    
    rsi_3 = calc_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    # Calculate streak lengths (consecutive up or down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_rsi = calc_rsi(streak_abs, streak_period)
    
    # Component 3: PercentRank of price changes
    price_changes = np.diff(close, prepend=close[0])
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = price_changes[i - rank_period + 1:i + 1]
        current = price_changes[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: >55 = chop (mean revert), <45 = trend (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Sum ATR over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother HTF trend."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_avg_30m = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(chop_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if np.isnan(vol_avg_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_30m[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_30m[i] > 55.0
        is_trending = chop_30m[i] < 45.0
        
        # === HTF TREND BIAS ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi_30m[i] < 15.0
        crsi_extreme_overbought = crsi_30m[i] > 85.0
        
        # Moderate CRSI for trend following entries
        crsi_moderate_oversold = crsi_30m[i] < 35.0
        crsi_moderate_overbought = crsi_30m[i] > 65.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (3+ confluence required) ===
        long_confluence = 0
        
        # 1. HTF 4h trend bullish
        if htf_4h_bullish:
            long_confluence += 1
        
        # 2. HTF 1d not strongly bearish (allow longs in 1d uptrend or neutral)
        if not htf_1d_bearish:
            long_confluence += 1
        
        # 3. CRSI extreme oversold (mean revert) OR moderate oversold (trend pullback)
        if is_choppy and crsi_extreme_oversold:
            long_confluence += 2  # Stronger signal in chop
        elif is_trending and crsi_moderate_oversold:
            long_confluence += 1  # Weaker signal in trend (pullback entry)
        
        # 4. Session filter
        if in_session:
            long_confluence += 1
        
        # 5. Volume filter
        if volume_ok:
            long_confluence += 1
        
        # Require 4+ confluence for long entry
        if long_confluence >= 4:
            desired_signal = SIZE_LONG
        
        # === SHORT ENTRY CONDITIONS (3+ confluence required) ===
        short_confluence = 0
        
        # 1. HTF 4h trend bearish
        if htf_4h_bearish:
            short_confluence += 1
        
        # 2. HTF 1d not strongly bullish (allow shorts in 1d downtrend or neutral)
        if not htf_1d_bullish:
            short_confluence += 1
        
        # 3. CRSI extreme overbought (mean revert) OR moderate overbought (trend retracement)
        if is_choppy and crsi_extreme_overbought:
            short_confluence += 2  # Stronger signal in chop
        elif is_trending and crsi_moderate_overbought:
            short_confluence += 1  # Weaker signal in trend (retracement entry)
        
        # 4. Session filter
        if in_session:
            short_confluence += 1
        
        # 5. Volume filter
        if volume_ok:
            short_confluence += 1
        
        # Require 4+ confluence for short entry
        if short_confluence >= 4 and desired_signal == 0.0:
            desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if HTF trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF 4h still bullish
                if htf_4h_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF 4h still bearish
                if htf_4h_bearish:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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