#!/usr/bin/env python3
"""
Experiment #222: 12h Primary + 1d/1w HTF — KAMA Trend + Choppiness Regime + Connors RSI

Hypothesis: After 12h failures with HMA+Donchian (#216), switch to research-proven patterns:
1. Connors RSI (CRSI) for mean reversion - 75% win rate in research, worked on ETH (Sharpe +0.923)
2. Choppiness Index for regime detection - CHOP>61.8=range (mean revert), CHOP<38.2=trend
3. KAMA for adaptive trend - adjusts to volatility, better than HMA in choppy markets
4. Dual regime logic: mean revert in chop, trend follow otherwise
5. LOOSER entry thresholds to ensure ≥30 trades/year (critical after #216 failures)

Key differences from #216:
- CRSI instead of simple RSI (more sensitive to reversals)
- Choppiness regime filter (switch logic based on market state)
- KAMA instead of HMA (adaptive to volatility)
- Lower CRSI thresholds (15/85 instead of 10/90) for more trades
- 1w HMA for ultra-long-term bias filter

TARGET: 25-40 trades/year, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.15, ±0.30 (discrete, max 0.35)
Stoploss: ATR(14) 2.5x trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_crsi_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average (KAMA).
    Adapts to market volatility - moves fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
    
    # Smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Mean reversion indicator with 75% win rate in research.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
    
    # RSI of streak
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent Rank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # CRSI
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi[:rank_period] = 50.0  # Fill initial values
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period:i+1])
        lowest_low = np.min(low[i-period:i+1])
        
        if highest_high > lowest_low:
            atr_sum = np.sum(calculate_atr(high[i-period:i+1], low[i-period:i+1], close[i-period:i+1], 1))
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    kama_12h = calculate_kama(close, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 1d HMA for medium-term bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_12h[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market - mean revert
        is_trending = chop[i] < 45.0  # Trend market - trend follow
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === CRSI EXTREMES (Mean Reversion) ===
        crsi_oversold = crsi[i] < 20.0  # Long signal
        crsi_overbought = crsi[i] > 80.0  # Short signal
        crsi_neutral = 30.0 <= crsi[i] <= 70.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # REGIME 1: CHOPPY MARKET - Mean Reversion
        if is_choppy:
            # Long: CRSI oversold + price above 1w HMA (long-term bull bias)
            if crsi_oversold:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_FULL
                elif price_above_hma_1d:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: CRSI overbought + price below 1w HMA (long-term bear bias)
            elif crsi_overbought:
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_FULL
                elif price_below_hma_1d:
                    new_signal = -POSITION_SIZE_HALF
        
        # REGIME 2: TRENDING MARKET - Trend Following
        elif is_trending:
            # Long: KAMA bullish + CRSI not overbought + 1d bias
            if kama_bullish and crsi[i] < 75.0:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: KAMA bearish + CRSI not oversold + 1d bias
            elif kama_bearish and crsi[i] > 25.0:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        # REGIME 3: TRANSITION - Conservative positions
        else:
            # Only take trades with strong HTF alignment
            if crsi_oversold and price_above_hma_1w and kama_bullish:
                new_signal = POSITION_SIZE_HALF
            elif crsi_overbought and price_below_hma_1w and kama_bearish:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if KAMA still bullish or CRSI not overbought
                if kama_bullish or crsi[i] < 85.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if KAMA still bearish or CRSI not oversold
                if kama_bearish or crsi[i] > 15.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if KAMA crosses bearish in trending regime
        if in_position and position_side > 0 and is_trending and kama_bearish:
            new_signal = 0.0
        
        # Exit short if KAMA crosses bullish in trending regime
        if in_position and position_side < 0 and is_trending and kama_bullish:
            new_signal = 0.0
        
        # Exit if CRSI reaches extreme opposite
        if in_position and position_side > 0 and crsi_overbought:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi_oversold:
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
                # Position flip
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