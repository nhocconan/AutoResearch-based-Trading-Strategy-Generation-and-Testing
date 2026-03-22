#!/usr/bin/env python3
"""
Experiment #531: 4h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 476 failed strategies (mostly volspike/Fisher combos), use a REGIME-ADAPTIVE
approach that switches between mean-reversion (in chop) and trend-following (in trends).

Key insights from research:
- Choppiness Index is the BEST meta-filter for bear/range markets (research note #3)
- Connors RSI has 75% win rate in mean-reversion regimes
- HMA reduces lag vs EMA for trend entries
- 1w HTF prevents counter-secular trades (major failure mode in 2022)

This strategy uses:
1. Choppiness Index(14) regime detection: CHOP>61.8=range, CHOP<38.2=trend
2. Connors RSI for mean-reversion entries in range regime
3. HMA(16/48) crossover for trend entries in trend regime
4. 1d HMA(21) for intermediate trend direction
5. 1w HMA(50) for secular trend filter (avoid counter-secular)
6. ATR(14) 3.0x trailing stop (wider to reduce whipsaw)
7. Discrete position sizing (0.25-0.30) based on regime confidence

Why this might work:
- Regime-adaptive = works in both bull/bear/range markets
- Connors RSI proven in bear markets (2022 crash, 2025 range)
- HMA trend following works in strong trends (2021 bull, SOL rallies)
- 1w filter prevents catastrophic counter-secular trades
- 4h TF targets 25-40 trades/year (optimal fee/trade ratio)

Position sizing: 0.25 (range), 0.30 (trend with confirmation)
Stoploss: 3.0 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_connors_hma_1d1w_v1"
timeframe = "4h"
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
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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
    Calculate Connors RSI (CRSI) - proven 75% win rate for mean reversion.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close - short-term momentum
    2. RSI(2) of streak - consecutive up/down days
    3. PercentRank(100) - where current price ranks vs last 100 bars
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) of streak
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
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP > 61.8 = range/choppy (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr_vals = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr_vals[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF HMA for intermediate trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 1w HTF HMA for secular trend
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # HMA for trend entries
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    
    # Connors RSI for mean-reversion entries
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Choppiness Index for regime detection
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Standard RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_RANGE = 0.25  # Mean reversion regime
    SIZE_TREND = 0.30  # Trend regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track HMA crossover
    prev_hma_16 = np.zeros(n)
    prev_hma_48 = np.zeros(n)
    prev_hma_16[1:] = hma_4h_16[:-1]
    prev_hma_48[1:] = hma_4h_48[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_range = chop[i] > 61.8  # Range/choppy regime - mean reversion
        chop_trend = chop[i] < 38.2  # Trending regime - trend follow
        chop_neutral = not chop_range and not chop_trend  # Transition - be cautious
        
        # === 1W SECULAR TREND (avoid counter-secular trades) ===
        secular_bull = close[i] > hma_1w_50_aligned[i]
        secular_bear = close[i] < hma_1w_50_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        bull_slope = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_slope = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H HMA CROSSOVER SIGNALS ===
        hma_cross_up = (hma_4h_16[i] > hma_4h_48[i]) and (prev_hma_16[i] <= prev_hma_48[i])
        hma_cross_down = (hma_4h_16[i] < hma_4h_48[i]) and (prev_hma_16[i] >= prev_hma_48[i])
        hma_aligned_bull = hma_4h_16[i] > hma_4h_48[i]
        hma_aligned_bear = hma_4h_16[i] < hma_4h_48[i]
        
        # === CONNORS RSI EXTREMES (mean reversion) ===
        crsi_oversold = crsi[i] < 10.0  # Strong long signal
        crsi_overbought = crsi[i] > 90.0  # Strong short signal
        crsi_mild_oversold = crsi[i] < 20.0
        crsi_mild_overbought = crsi[i] > 80.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # RANGE REGIME (mean reversion with Connors RSI)
        if chop_range:
            # Long: CRSI oversold + above 1w HMA (secular bull bias)
            if crsi_oversold and secular_bull:
                new_signal = SIZE_RANGE
            elif crsi_mild_oversold and secular_bull and bull_regime:
                new_signal = SIZE_RANGE * 0.8
            # Short: CRSI overbought + below 1w HMA (secular bear bias)
            elif crsi_overbought and secular_bear:
                new_signal = -SIZE_RANGE
            elif crsi_mild_overbought and secular_bear and bear_regime:
                new_signal = -SIZE_RANGE * 0.8
        
        # TREND REGIME (trend following with HMA crossover)
        elif chop_trend:
            # Long: HMA cross up + bull regime + secular bull
            if hma_cross_up and bull_regime and secular_bull:
                new_signal = SIZE_TREND
            # Long pullback: HMA aligned bull + RSI pullback + secular bull
            elif hma_aligned_bull and bull_regime and rsi_14[i] < 45.0 and secular_bull:
                new_signal = SIZE_TREND * 0.8
            # Short: HMA cross down + bear regime + secular bear
            elif hma_cross_down and bear_regime and secular_bear:
                new_signal = -SIZE_TREND
            # Short bounce: HMA aligned bear + RSI bounce + secular bear
            elif hma_aligned_bear and bear_regime and rsi_14[i] > 55.0 and secular_bear:
                new_signal = -SIZE_TREND * 0.8
        
        # NEUTRAL REGIME (transition - only take strongest signals)
        elif chop_neutral:
            # Only take extreme CRSI signals with strong HTF alignment
            if crsi_oversold and secular_bull and bull_regime and bull_slope:
                new_signal = SIZE_RANGE * 0.6
            elif crsi_overbought and secular_bear and bear_regime and bear_slope:
                new_signal = -SIZE_RANGE * 0.6
        
        # === STOPLOSS CHECK (3.0 * ATR trailing - wider to reduce whipsaw) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or extreme) ===
        # Exit long on secular flip or extreme CRSI
        if in_position and position_side > 0:
            if secular_bear and bear_regime:  # Major regime flip
                new_signal = 0.0
            elif crsi[i] > 95.0:  # Extreme overbought
                new_signal = 0.0
        
        # Exit short on secular flip or extreme CRSI
        if in_position and position_side < 0:
            if secular_bull and bull_regime:  # Major regime flip
                new_signal = 0.0
            elif crsi[i] < 5.0:  # Extreme oversold
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