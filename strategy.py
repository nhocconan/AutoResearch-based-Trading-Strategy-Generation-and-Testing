#!/usr/bin/env python3
"""
Experiment #285: 1h Primary + 4h/1d HTF — Regime-Adaptive Connors RSI + Session Filter

Hypothesis: After 258 failed experiments, the key insight is that 1h strategies fail
due to TOO MANY TRADES (>200/year) → fee drag destroys profits. This strategy uses:

1. 1d HMA(21) for PRIMARY trend direction (very slow, few regime changes)
2. 4h Choppiness(14) for regime detection (trend vs mean-revert mode)
3. 1h Connors RSI for entry timing (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. Session filter: ONLY trade 8-20 UTC (highest liquidity, lowest slippage)
5. Volume filter: ONLY trade when volume > 0.8x 20-bar average
6. ATR(14) trailing stoploss at 2.5x

Key innovation: 1d trend sets direction, 4h chop sets mode, 1h CRSI times entry.
This gives ~40-60 trades/year (appropriate for 1h) with HTF-level discipline.

Position sizing: 0.20 base, 0.35 strong conviction (discrete levels)
Target: 30-60 trades/year per symbol (CRITICAL for 1h to avoid fee drag)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_chop_session_vol_4h1d_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank of returns
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i].dropna()
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[:i].dropna() < returns.iloc[i]).sum() / len(window) * 100
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    # Handle NaN
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_session_filter(open_time):
    """
    Filter for high-liquidity session (8-20 UTC).
    Returns boolean array: True = trade allowed, False = skip
    """
    # open_time is in milliseconds since epoch
    hours = pd.to_datetime(open_time, unit='ms').dt.hour.values
    session_active = (hours >= 8) & (hours <= 20)
    return session_active

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
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 4h HTF indicators (regime detection)
    chop_4h_14 = calculate_choppiness_index(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    chop_4h_14_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    session_active = calculate_session_filter(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(chop_4h_14_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        # Bull: price above 1d HMA(21) AND HMA(21) > HMA(50)
        # Bear: price below 1d HMA(21) AND HMA(21) < HMA(50)
        regime_bull = (close[i] > hma_1d_21_aligned[i]) and (hma_1d_21_aligned[i] > hma_1d_50_aligned[i])
        regime_bear = (close[i] < hma_1d_21_aligned[i]) and (hma_1d_21_aligned[i] < hma_1d_50_aligned[i])
        regime_neutral = not regime_bull and not regime_bear
        
        # === 4H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (pullback entries in trend direction)
        is_choppy = chop_4h_14_aligned[i] > 55.0
        is_trending = chop_4h_14_aligned[i] < 45.0
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = session_active[i]
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / vol_avg_20[i]
        volume_ok = vol_ratio > 0.8
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_oversold = crsi[i] < 25.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_overbought = crsi[i] > 75.0
        
        # === ENTRY LOGIC (3+ confluence required) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG entries (need: session + volume + 3+ indicators)
        long_confluence = 0
        if in_session:
            long_confluence += 1
        if volume_ok:
            long_confluence += 1
        if regime_bull or regime_neutral:
            long_confluence += 1
        if crsi_extreme_oversold:
            long_confluence += 2
        elif crsi_oversold:
            long_confluence += 1
        if is_choppy:
            long_confluence += 1  # mean revert mode favors oversold
        
        # SHORT entries (need: session + volume + 3+ indicators)
        short_confluence = 0
        if in_session:
            short_confluence += 1
        if volume_ok:
            short_confluence += 1
        if regime_bear or regime_neutral:
            short_confluence += 1
        if crsi_extreme_overbought:
            short_confluence += 2
        elif crsi_overbought:
            short_confluence += 1
        if is_choppy:
            short_confluence += 1  # mean revert mode favors overbought
        
        # TREND MODE: only enter in trend direction with pullback
        if is_trending and not is_choppy:
            if regime_bull and long_confluence >= 4 and crsi_oversold:
                new_signal = BASE_SIZE
            if regime_bull and long_confluence >= 5 and crsi_extreme_oversold:
                new_signal = STRONG_SIZE
            if regime_bear and short_confluence >= 4 and crsi_overbought:
                new_signal = -BASE_SIZE
            if regime_bear and short_confluence >= 5 and crsi_extreme_overbought:
                new_signal = -STRONG_SIZE
        
        # RANGE MODE: mean revert at extremes (both directions allowed)
        if is_choppy:
            if long_confluence >= 4 and crsi_extreme_oversold:
                new_signal = BASE_SIZE
            if long_confluence >= 5 and crsi_extreme_oversold and regime_bull:
                new_signal = STRONG_SIZE
            if short_confluence >= 4 and crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            if short_confluence >= 5 and crsi_extreme_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # Force trade if no signal for 30 bars (~30h = 1.25 days on 1h)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 35 and in_session and volume_ok:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and crsi[i] > 65 and in_session and volume_ok:
                new_signal = -BASE_SIZE * 0.8
            elif is_choppy and crsi[i] < 20 and in_session:
                new_signal = BASE_SIZE * 0.7
            elif is_choppy and crsi[i] > 80 and in_session:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and chop_4h_14_aligned[i] < 45:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and chop_4h_14_aligned[i] < 45:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals