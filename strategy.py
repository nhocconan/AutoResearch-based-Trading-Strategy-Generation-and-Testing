#!/usr/bin/env python3
"""
Experiment #503: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 499 failed experiments, return to proven higher-timeframe patterns.
Research shows 1d primary with regime detection works best for BTC/ETH through 2022 crash.

Key Components:
1. CHOPPINESS INDEX (14): Regime detection — CHOP>55 = range (mean revert), CHOP<45 = trend
   This is the META-FILTER that determines which strategy to use
2. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Proven 75% win rate on mean reversion. Long CRSI<20, Short CRSI>80
3. 1w HMA(21): Major trend direction filter — only trade with weekly trend
4. DUAL REGIME: Mean revert in chop, trend-follow in trends (asymmetric)
5. ATR(14) trailing stop: 2.5x for protection

Why this might beat current best (Sharpe=0.435):
- 1d primary = proven to work (exp#497 had Sharpe=0.093, simpler = better)
- Choppiness regime switch = adapts to 2022 crash AND 2021 bull
- CRSI = different from standard RSI (448 strategies failed with standard RSI)
- 1w HTF = major trend filter prevents counter-trend trades
- Fewer conflicting filters = more trades (critical: need >=30/symbol)

Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 25-40 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_hma_1w_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    Values between = transitional
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    highest = high_s.rolling(window=period, min_periods=period).max()
    lowest = low_s.rolling(window=period, min_periods=period).min()
    
    # ATR
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    
    # CHOP formula
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, 1e-10)
    
    chop = 100.0 * np.log10((atr_s * period) / range_hl) / np.log10(period)
    chop = chop.fillna(50.0)  # Default to neutral
    
    return chop.values

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Where current price ranks vs last 100 days
    
    Long: CRSI < 20 (oversold)
    Short: CRSI > 80 (overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    avg_loss = loss.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=period_streak, min_periods=period_streak, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank (where current close ranks vs last N days)
    percent_rank = close_s.rolling(window=period_rank, min_periods=period_rank).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100.0 if len(x) > 1 else 50.0,
        raw=False
    )
    
    # Combine components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    crsi = crsi.fillna(50.0)
    
    return crsi.values

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

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
    chop = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
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
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        choppy_regime = chop[i] > 55.0  # Range market → mean reversion
        trending_regime = chop[i] < 45.0  # Trending market → trend follow
        # Between 45-55 = transitional (use both)
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 25.0  # Relaxed from 20 for more trades
        crsi_overbought = crsi[i] > 75.0  # Relaxed from 80 for more trades
        crsi_extreme_low = crsi[i] < 15.0
        crsi_extreme_high = crsi[i] > 85.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # === MEAN REVERSION MODE (Choppy regime) ===
        if choppy_regime:
            # Long: CRSI oversold + above weekly HMA or above SMA200
            if crsi_oversold and (bull_regime or above_sma200):
                new_signal = LONG_SIZE
            # Extra strong long: CRSI extreme + RSI oversold
            elif crsi_extreme_low and rsi_oversold:
                new_signal = LONG_SIZE
            # Short: CRSI overbought + below weekly HMA or below SMA200
            elif crsi_overbought and (bear_regime or below_sma200):
                new_signal = -SHORT_SIZE
            # Extra strong short: CRSI extreme + RSI overbought
            elif crsi_extreme_high and rsi_overbought:
                new_signal = -SHORT_SIZE
        
        # === TREND FOLLOW MODE (Trending regime) ===
        elif trending_regime:
            # Long: Bull regime + CRSI pullback (not extreme oversold)
            if bull_regime and crsi[i] < 50.0 and rsi_oversold:
                new_signal = LONG_SIZE
            # Short: Bear regime + CRSI bounce (not extreme overbought)
            elif bear_regime and crsi[i] > 50.0 and rsi_overbought:
                new_signal = -SHORT_SIZE
        
        # === TRANSITIONAL MODE (45-55 CHOP) ===
        else:
            # Conservative: only extreme CRSI signals
            if crsi_extreme_low and bull_regime:
                new_signal = LONG_SIZE * 0.7
            elif crsi_extreme_high and bear_regime:
                new_signal = -SHORT_SIZE * 0.7
        
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
        
        # === EXIT CONDITIONS ===
        # Exit long on CRSI overbought or regime flip bearish
        if in_position and position_side > 0:
            if crsi_overbought or (bear_regime and chop[i] < 45.0):
                new_signal = 0.0
        
        # Exit short on CRSI oversold or regime flip bullish
        if in_position and position_side < 0:
            if crsi_oversold or (bull_regime and chop[i] < 45.0):
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
                # Flip position
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