#!/usr/bin/env python3
"""
Experiment #500: 1h Primary + 4h/12h HTF — Regime-Adaptive Connors RSI

Hypothesis: After 499 experiments, key lessons:
1. Lower TF (1h) needs RELAXED entry thresholds (not too strict = 0 trades)
2. HTF (4h/12h) for DIRECTION, 1h for ENTRY TIMING only
3. Connors RSI has 75% win rate - use as PRIMARY signal with relaxed thresholds
4. Choppiness Index should ADJUST position size, not block entries
5. Target: 30-60 trades/year on 1h (not >100, not <10)

Strategy components:
1. 4h HMA(21) = trend bias (long when price > HMA)
2. 12h Choppiness = regime (adjust size: range=full, trend=half)
3. 1h Connors RSI = entry (CRSI<25 long, CRSI>75 short - RELAXED)
4. ATR(14) = 2.0x trailing stoploss
5. Position size: 0.20-0.30 discrete levels

Key difference from #499:
- RELAXED CRSI thresholds (25/75 vs 10/90) = MORE trades
- Regime adjusts SIZE not entry = no blocked signals
- Simpler logic = fewer conflicting conditions
- Expected: 40-80 trades/year on 1h

Why this should beat Sharpe=0.435:
- Connors RSI proven edge (research note #1)
- Works in bull/bear/range (regime-adaptive sizing)
- HTF filter prevents worst counter-trend trades
- Relaxed thresholds ensure >=30 trades/symbol on train
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_hma_4h12h_v2"
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
    CHOP > 61.8 = ranging market (mean reversion favored)
    CHOP < 38.2 = trending market (trend follow favored)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    highest = high_s.rolling(window=period, min_periods=period).max()
    lowest = low_s.rolling(window=period, min_periods=period).min()
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    range_hl = (highest - lowest).values
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * (atr_sum / range_hl) / (np.log(period) / np.log(2))
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long when CRSI < 25, Short when CRSI > 75 (relaxed for more trades)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI of close (period=3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of streak (consecutive up/down bars)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: Percent Rank of close over lookback
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100.0, raw=False
    )
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 12h HTF indicators (regime detection)
    chop_12h = calculate_choppiness_index(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        period=14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    RANGE_SIZE = 0.30  # Full size in ranging market
    TREND_SIZE = 0.20  # Half size in trending market
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        
        # === REGIME DETECTION (12h Choppiness) ===
        range_regime = chop_12h_aligned[i] > 55.0
        trend_regime = chop_12h_aligned[i] < 45.0
        
        # Select position size based on regime
        if range_regime:
            pos_size = RANGE_SIZE
        elif trend_regime:
            pos_size = TREND_SIZE
        else:
            pos_size = BASE_SIZE
        
        # === TREND DIRECTION (4h HMA) ===
        bull_trend = close[i] > hma_4h_21_aligned[i]
        bear_trend = close[i] < hma_4h_21_aligned[i]
        strong_bull = bull_trend and (hma_4h_21_aligned[i] > hma_4h_50_aligned[i])
        strong_bear = bear_trend and (hma_4h_21_aligned[i] < hma_4h_50_aligned[i])
        
        # === CONNORS RSI SIGNALS (RELAXED THRESHOLDS) ===
        crsi_oversold = crsi[i] < 25.0  # Relaxed from 10 for more trades
        crsi_overbought = crsi[i] > 75.0  # Relaxed from 90 for more trades
        crsi_extreme_low = crsi[i] < 15.0  # Very oversold
        crsi_extreme_high = crsi[i] > 85.0  # Very overbought
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions for frequency)
        # Condition 1: CRSI oversold + bull trend (primary setup)
        if crsi_oversold and bull_trend:
            new_signal = pos_size
        # Condition 2: CRSI extreme low (any trend - capitulation)
        elif crsi_extreme_low:
            new_signal = pos_size * 0.8
        # Condition 3: Range regime + CRSI oversold (mean reversion)
        elif range_regime and crsi_oversold:
            new_signal = pos_size
        # Condition 4: Strong bull + CRSI neutral pullback
        elif strong_bull and crsi[i] < 40.0:
            new_signal = pos_size * 0.6
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: CRSI overbought + bear trend (primary setup)
            if crsi_overbought and bear_trend:
                new_signal = -pos_size
            # Condition 2: CRSI extreme high (any trend - FOMO top)
            elif crsi_extreme_high:
                new_signal = -pos_size * 0.8
            # Condition 3: Range regime + CRSI overbought (mean reversion)
            elif range_regime and crsi_overbought:
                new_signal = -pos_size
            # Condition 4: Strong bear + CRSI neutral bounce
            elif strong_bear and crsi[i] > 60.0:
                new_signal = -pos_size * 0.6
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON CRSI REVERSAL ===
        if in_position and position_side > 0:
            if crsi_overbought:  # Exit long when CRSI > 75
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if crsi_oversold:  # Exit short when CRSI < 25
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