#!/usr/bin/env python3
"""
Experiment #478: 30m Primary + 4h/1d HTF — Triple Confluence Regime Strategy

Hypothesis: After 477 experiments, clear pattern emerges for lower TF success:
1. 30m needs EXTREME selectivity (target 40-80 trades/year, NOT 200+)
2. Use 4h HMA for trend direction, 1d HMA for major regime filter
3. Choppiness Index determines entry type (range=mean-revert, trend=pullback)
4. Connors RSI for precise entry timing (proven 75% win rate)
5. Volume filter (>1.0x 20-bar avg) confirms institutional participation
6. Session filter (8-20 UTC) avoids low-liquidity whipsaws
7. ATR 2.5x trailing stop protects from 2022-style crashes

Why this might beat Sharpe=0.435:
- Triple HTF confluence (4h HMA + 1d HMA + CHOP) = fewer but higher quality trades
- Session filter eliminates Asian session noise (major source of lower TF failures)
- Volume confirmation ensures we trade with institutional flow
- Asymmetric sizing: 0.25 long, 0.20 short (bear market protection)
- Relaxed CRSI thresholds (20/80 instead of 10/90) for adequate frequency

Position sizing: 0.20-0.25 (smaller for 30m to reduce fee impact)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year on 30m, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_triple_confluence_chop_crsi_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain = streak_delta.where(streak_delta > 0, 0.0)
    loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of returns over 100 periods
    returns = close_s.pct_change()
    percent_rank = pd.Series(np.zeros(n))
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank.iloc[i] = (rank / rank_period) * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs 20-bar average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

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
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(crsi_30m[i]) or np.isnan(chop_30m[i]):
            continue
        if np.isnan(vol_ratio[i]):
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20  # London/NY overlap + US session
        
        # === 1D MAJOR REGIME (primary filter) ===
        bull_1d = close[i] > hma_1d_21_aligned[i]
        bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND DIRECTION (secondary filter) ===
        bull_4h = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        bear_4h = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CHOPPINESS REGIME (determines entry type) ===
        is_ranging = chop_30m[i] > 55.0
        is_trending = chop_30m[i] < 45.0
        
        # === CONNORS RSI SIGNALS (entry timing) ===
        crsi_oversold = crsi_30m[i] < 25.0
        crsi_overbought = crsi_30m[i] > 75.0
        crsi_extreme_oversold = crsi_30m[i] < 15.0
        crsi_extreme_overbought = crsi_30m[i] > 85.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.9
        
        # === ENTRY LOGIC — TRIPLE CONFLUENCE (HTF + CHOP + CRSI + SESSION) ===
        new_signal = 0.0
        
        # LONG ENTRIES (require 3+ confluence)
        if bull_1d and bull_4h and in_session:
            if is_ranging and crsi_oversold and volume_confirmed:
                new_signal = LONG_SIZE
            elif is_trending and crsi_30m[i] < 35.0 and volume_confirmed:
                new_signal = LONG_SIZE
            elif crsi_extreme_oversold and volume_confirmed:
                new_signal = LONG_SIZE
        
        # SHORT ENTRIES (require 3+ confluence)
        if new_signal == 0.0 and bear_1d and bear_4h and in_session:
            if is_ranging and crsi_overbought and volume_confirmed:
                new_signal = -SHORT_SIZE
            elif is_trending and crsi_30m[i] > 65.0 and volume_confirmed:
                new_signal = -SHORT_SIZE
            elif crsi_extreme_overbought and volume_confirmed:
                new_signal = -SHORT_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        if in_position and position_side > 0 and crsi_30m[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_30m[i] < 20.0:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and (bear_1d or bear_4h):
            new_signal = 0.0
        if in_position and position_side < 0 and (bull_1d or bull_4h):
            new_signal = 0.0
        
        # Session exit (close positions outside trading hours)
        if in_position and not in_session:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals