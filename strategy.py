#!/usr/bin/env python3
"""
Experiment #212: 12h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI + HMA)

Hypothesis: 12h timeframe with dual regime switching based on Choppiness Index.
In choppy markets (CHOP > 61.8): mean reversion using Connors RSI extremes.
In trending markets (CHOP < 38.2): trend following using HMA crossover + RSI pullback.
Filtered by 1d/1w macro bias for direction confirmation.

Why this should work:
1. 12h has proven success in past experiments (#206 had Sharpe=0.024, needs improvement)
2. Choppiness Index is the BEST regime filter for bear/range markets (research-backed)
3. Connors RSI has 75% win rate on mean reversion entries
4. Dual regime adapts to market conditions instead of forcing one style
5. 1d/1w HTF provides macro bias to avoid counter-trend trades

TARGET: 20-50 trades/year on 12h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_dual_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    RSI(Streak): RSI of consecutive up/down days
    PercentRank: percentage of past returns lower than current return
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0)
    
    # Streak RSI
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
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0)
    
    # Percent Rank
    percent_rank = np.zeros(n)
    returns = close_s.pct_change()
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period+1:i+1].dropna()
        if len(window) > 0:
            current_return = returns.iloc[i]
            if not np.isnan(current_return):
                percent_rank[i] = 100.0 * (window < current_return).sum() / len(window)
    
    crsi = (rsi_close.values + rsi_streak.values + percent_rank) / 3.0
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_atr = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_atr = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * plus_atr[i] / atr[i]
            minus_di[i] = 100.0 * minus_atr[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for long-term bias (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
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
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(chop[i]):
            continue
        
        # === HTF MACRO BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        # Neutral zone: 45-55 (use trend logic with smaller size)
        
        # === TREND DETECTION (12h HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_14[i] > 25.0
        weak_trend = adx_14[i] < 20.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- CHOPPY REGIME: Mean Reversion with Connors RSI ---
        if is_choppy:
            # LONG: CRSI < 15 (oversold) + price > 1d HMA (bullish bias)
            if crsi[i] < 15.0:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL
                elif not price_below_hma_1w:  # At least not bearish on weekly
                    new_signal = POSITION_SIZE_HALF
            
            # SHORT: CRSI > 85 (overbought) + price < 1d HMA (bearish bias)
            elif crsi[i] > 85.0:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL
                elif not price_above_hma_1w:  # At least not bullish on weekly
                    new_signal = -POSITION_SIZE_HALF
        
        # --- TRENDING REGIME: HMA Crossover + RSI Pullback ---
        elif is_trending:
            # LONG: HMA bullish + RSI pullback (35-55) + ADX confirms
            if hma_bullish:
                if rsi_14[i] >= 35.0 and rsi_14[i] <= 55.0:
                    if price_above_hma_1d:
                        new_signal = POSITION_SIZE_FULL
                    elif strong_trend:
                        new_signal = POSITION_SIZE_HALF
            
            # SHORT: HMA bearish + RSI pullback (45-65) + ADX confirms
            elif hma_bearish:
                if rsi_14[i] >= 45.0 and rsi_14[i] <= 65.0:
                    if price_below_hma_1d:
                        new_signal = -POSITION_SIZE_FULL
                    elif strong_trend:
                        new_signal = -POSITION_SIZE_HALF
        
        # --- NEUTRAL ZONE: Conservative trend following ---
        else:
            if hma_bullish and rsi_14[i] >= 40.0 and rsi_14[i] <= 55.0:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_HALF
            elif hma_bearish and rsi_14[i] >= 45.0 and rsi_14[i] <= 60.0:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if HMA still bullish or CRSI not overbought
                if (hma_bullish and rsi_14[i] < 70.0) or (is_choppy and crsi[i] < 80.0):
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if HMA still bearish or CRSI not oversold
                if (hma_bearish and rsi_14[i] > 30.0) or (is_choppy and crsi[i] > 20.0):
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
        # Exit long if HMA crosses bearish (in trending regime)
        if in_position and position_side > 0 and is_trending and hma_bearish:
            new_signal = 0.0
        
        # Exit short if HMA crosses bullish (in trending regime)
        if in_position and position_side < 0 and is_trending and hma_bullish:
            new_signal = 0.0
        
        # Exit if macro trend reverses against position
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
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