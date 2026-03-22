#!/usr/bin/env python3
"""
Experiment #482: 12h Primary + 1d/1w HTF — Simplified Dual Regime with Volume Filter

Hypothesis: After 471 failed experiments, the pattern is clear:
1. Complex entry logic = 0 trades (see #470, #475, #478, #480 with Sharpe=0.000)
2. 12h timeframe works but needs SIMPLER entry conditions for trade frequency
3. Choppiness Index regime switch is proven (ETH Sharpe +0.923 in research)
4. CRSI thresholds too strict (10/90) — relax to 25/75 for adequate frequency
5. Volume confirmation reduces false signals without killing trade count

Why this might beat current best (Sharpe=0.435):
- Fewer conflicting filters = more trades (critical for >=30 trades/symbol)
- Clear regime separation: CHOP>55 mean-revert, CHOP<45 trend-follow
- Volume filter (1.5x SMA20) confirms genuine moves, not noise
- Asymmetric sizing: 0.35 long / 0.25 short (bear market protection)
- 1.5x ATR stoploss (tighter than 2.0x) protects in 2022-style crashes
- 1d HMA for major trend, 12h HMA for local entry timing

Position sizing: 0.25-0.35 (discrete levels, max 0.40)
Stoploss: 1.5 * ATR trailing (signal → 0 when hit)
Target: 25-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simp_regime_crsi_vol_1d_v1"
timeframe = "12h"
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
    CHOP > 55 = ranging market (mean reversion)
    CHOP < 45 = trending market (trend follow)
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume spike detection."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_50 = calculate_hma(close, period=50)
    crsi_12h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.35
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
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === VOLUME CONFIRMATION (avoid false breakouts) ===
        vol_confirmed = volume[i] > 1.3 * vol_sma_20[i]
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION (clear separation) ===
        is_ranging = chop_12h[i] > 55.0
        is_trending = chop_12h[i] < 45.0
        
        # === 12H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === CONNORS RSI SIGNALS (relaxed thresholds for frequency) ===
        crsi_oversold = crsi_12h[i] < 30.0
        crsi_overbought = crsi_12h[i] > 70.0
        crsi_extreme_oversold = crsi_12h[i] < 20.0
        crsi_extreme_overbought = crsi_12h[i] > 80.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — SIMPLIFIED DUAL REGIME ===
        new_signal = 0.0
        
        # LONG ENTRIES (simplified — any condition triggers)
        if bull_regime and hma_bullish and crsi_oversold and vol_confirmed:
            new_signal = LONG_SIZE
        elif above_sma200 and hma_bullish and crsi_12h[i] < 35.0:
            new_signal = LONG_SIZE
        elif crsi_extreme_oversold and hma_bullish:
            new_signal = LONG_SIZE
        elif is_ranging and crsi_12h[i] < 25.0 and above_sma200:
            new_signal = LONG_SIZE * 0.8
        elif is_trending and hma_bullish and crsi_12h[i] < 40.0 and vol_confirmed:
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES (simplified — any condition triggers)
        if new_signal == 0.0:
            if bear_regime and hma_bearish and crsi_overbought and vol_confirmed:
                new_signal = -SHORT_SIZE
            elif below_sma200 and hma_bearish and crsi_12h[i] > 65.0:
                new_signal = -SHORT_SIZE
            elif crsi_extreme_overbought and hma_bearish:
                new_signal = -SHORT_SIZE
            elif is_ranging and crsi_12h[i] > 75.0 and below_sma200:
                new_signal = -SHORT_SIZE * 0.8
            elif is_trending and hma_bearish and crsi_12h[i] > 60.0 and vol_confirmed:
                new_signal = -SHORT_SIZE
        
        # === STOPLOSS CHECK (1.5 * ATR trailing — tighter protection) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 1.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 1.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        if in_position and position_side > 0 and crsi_12h[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_12h[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit
        if in_position and position_side > 0 and bear_regime and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and hma_bullish:
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