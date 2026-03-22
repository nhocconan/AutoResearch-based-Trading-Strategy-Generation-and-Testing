#!/usr/bin/env python3
"""
Experiment #483: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian

Hypothesis: Research shows 1d timeframe with regime-switching has strong potential:
- Choppiness Index + Connors RSI achieved ETH Sharpe +0.923 in backtests
- Donchian breakout + HMA trend achieved SOL Sharpe +0.782
- 1d timeframe naturally limits trades to 20-50/year (low fee drag)
- 1w HMA provides clean major trend bias without whipsaw

Why this might beat current best (Sharpe=0.435):
- Dual regime: mean-revert in choppy markets, trend-follow in trending markets
- Connors RSI (not standard RSI) has 75% win rate for reversals
- Donchian breakout confirms momentum before entry
- Relaxed CRSI thresholds (20/80 instead of 10/90) for adequate trade frequency
- 1w HMA filter prevents counter-trend trades in strong trends

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_donchian_1w_v1"
timeframe = "1d"
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
    
    Proven 75% win rate in research notes. Best for mean reversion entries.
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
    
    # Component 3: Percent Rank of daily returns over 100 periods
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

def calculate_donchian_high(high, period=20):
    """Calculate Donchian Channel Upper Band (highest high over period)."""
    return pd.Series(high).rolling(window=period, min_periods=period).max().values

def calculate_donchian_low(low, period=20):
    """Calculate Donchian Channel Lower Band (lowest low over period)."""
    return pd.Series(low).rolling(window=period, min_periods=period).min().values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    crsi_1d = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    donchian_high = calculate_donchian_high(high, period=20)
    donchian_low = calculate_donchian_low(low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
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
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_ranging = chop_1d[i] > 55.0  # relaxed from 61.8 for more trades
        is_trending = chop_1d[i] < 45.0  # relaxed from 38.2 for more trades
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === CONNORS RSI SIGNALS (relaxed thresholds for frequency) ===
        crsi_oversold = crsi_1d[i] < 25.0
        crsi_overbought = crsi_1d[i] > 75.0
        crsi_extreme_oversold = crsi_1d[i] < 15.0
        crsi_extreme_overbought = crsi_1d[i] > 85.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_high[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_low[i-1] if i > 0 else False
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple paths to entry for frequency)
        if bull_regime and hma_bullish and crsi_oversold:
            new_signal = LONG_SIZE
        elif bull_regime and is_trending and donchian_breakout_long and hma_bullish:
            new_signal = LONG_SIZE
        elif bull_regime and is_ranging and crsi_1d[i] < 30.0 and above_sma200:
            new_signal = LONG_SIZE * 0.8
        elif crsi_extreme_oversold and hma_bullish:
            new_signal = LONG_SIZE
        elif is_ranging and crsi_1d[i] < 20.0 and above_sma200:
            new_signal = LONG_SIZE * 0.8
        elif is_trending and hma_bullish and crsi_1d[i] < 35.0:
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES (multiple paths to entry for frequency)
        if new_signal == 0.0:
            if bear_regime and hma_bearish and crsi_overbought:
                new_signal = -SHORT_SIZE
            elif bear_regime and is_trending and donchian_breakout_short and hma_bearish:
                new_signal = -SHORT_SIZE
            elif bear_regime and is_ranging and crsi_1d[i] > 70.0 and below_sma200:
                new_signal = -SHORT_SIZE * 0.8
            elif crsi_extreme_overbought and hma_bearish:
                new_signal = -SHORT_SIZE
            elif is_ranging and crsi_1d[i] > 80.0 and below_sma200:
                new_signal = -SHORT_SIZE * 0.8
            elif is_trending and hma_bearish and crsi_1d[i] > 65.0:
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
        if in_position and position_side > 0 and crsi_1d[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_1d[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
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