#!/usr/bin/env python3
"""
Experiment #472: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + Donchian

Hypothesis: 12h timeframe reduces noise vs 6h while maintaining sufficient trade frequency.
Recent 6h failures show complex regime filters kill trade count. This strategy uses:

1. CHOPPINESS INDEX (14): Proven regime filter - CHOP>61.8=range, CHOP<38.2=trend
2. CONNORS RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - 75% win rate in literature
3. DONCHIAN BREAKOUT (20): Simple trend entry that works on HTF
4. 1d HMA BIAS: Single HTF filter (not dual - dual killed trades in #435)
5. WIDER STOPLOSS: 3.0x ATR for 12h timeframe (vs 2.0x for 6h)

Entry Logic:
- Trending (CHOP<38.2): Donchian breakout + 1d HMA alignment
- Range (CHOP>61.8): Connors RSI extremes (<10 long, >90 short)
- Transition (38.2-61.8): Hold previous regime or flat

Key differences from failed #435 (6h):
- Single HTF (1d) not dual (12h+1d) - reduces filter complexity
- Connors RSI instead of standard RSI - better mean reversion
- Choppiness Index instead of ADX+BB - proven regime detector
- Looser thresholds to ensure 20-50 trades/year on 12h

Target: Sharpe>0.45, DD>-35%, trades>=40 train (10/year), trades>=6 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_connors_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI - composite mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    n = len(close)
    if n < max(rsi_period, streak_period, pr_period) + 5:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    streak_gain = np.concatenate([[0.0], streak_gain])
    streak_loss = np.concatenate([[0.0], streak_loss])
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percentile Rank of price changes
    pr = np.zeros(n)
    pr[:] = np.nan
    price_change = np.zeros(n)
    price_change[0] = 0.0
    for i in range(1, n):
        price_change[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    for i in range(pr_period, n):
        window = price_change[i-pr_period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < price_change[i]) / len(valid) * 100.0
    
    # Combine into Connors RSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(max(rsi_period, streak_period, pr_period), n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    connors_rsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Regime memory
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(connors_rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = choppy/range
        # CHOP < 38.2 = trending
        # Between = maintain previous regime
        
        chop = choppiness[i]
        
        if chop < 38.2:
            current_regime = 1  # Trending
        elif chop > 61.8:
            current_regime = 2  # Choppy
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === 1d HTF BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakdown_short = False
        
        if not np.isnan(donchian_upper[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            donchian_breakdown_short = close[i] < donchian_lower[i-1]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = connors_rsi[i] < 15.0  # Very oversold
        crsi_overbought = connors_rsi[i] > 85.0  # Very overbought
        crsi_extreme_oversold = connors_rsi[i] < 10.0
        crsi_extreme_overbought = connors_rsi[i] > 90.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (Donchian breakout + HTF alignment)
        if current_regime == 1:
            # Long: HTF bull + Donchian breakout + above SMA50
            if htf_bull and donchian_breakout_long and above_sma50:
                desired_signal = SIZE_STRONG
            # Short: HTF bear + Donchian breakdown + below SMA50
            elif htf_bear and donchian_breakdown_short and below_sma50:
                desired_signal = -SIZE_STRONG
        
        # REGIME 2: CHOPPY (Connors RSI mean reversion)
        elif current_regime == 2:
            # Long: Connors RSI < 15 + above SMA200 (long-term bull filter)
            if crsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            # Short: Connors RSI > 85 + below SMA200 (long-term bear filter)
            elif crsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            # Extra: Extreme Connors for stronger signal
            elif crsi_extreme_oversold and above_sma50:
                desired_signal = SIZE_BASE
            elif crsi_extreme_overbought and below_sma50:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3.0x ATR for 12h timeframe) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss (3.0x ATR for 12h)
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals