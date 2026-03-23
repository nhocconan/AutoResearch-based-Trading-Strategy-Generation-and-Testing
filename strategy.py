#!/usr/bin/env python3
"""
Experiment #244: 4h Primary + 12h/1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 200+ failed experiments, the winning pattern combines:
1. Connors RSI (CRSI) for high-probability mean reversion entries (75% win rate in literature)
2. Choppiness Index to detect range vs trend regime
3. 12h HMA for intermediate trend direction
4. 1d HMA for macro bias alignment
5. ATR trailing stoploss for risk management

Key insight from failures: Overly complex regime switching fails. Simple confluence of 
3-4 proven signals works better. CRSI extremes ( <10 or >90) with trend alignment 
generates sufficient trades while maintaining quality.

TARGET: 25-50 trades/year on 4h, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_regime_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive days up (positive) or down (negative)
    PercentRank: percentile rank of today's return over last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term momentum
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - RSI of up/down streaks
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute values for RSI calculation
    streak_pos = np.maximum(streak, 0)
    streak_neg = np.maximum(-streak, 0)
    
    # RSI of streak (simplified - use streak direction as momentum)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        gains = np.maximum(streak[i-streak_period:i+1], 0)
        losses = np.maximum(-streak[i-streak_period:i+1], 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        if avg_loss == 0:
            streak_rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank - where does today's return rank in last 100?
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period:i+1].dropna()
        if len(window_returns) > 0:
            current_return = returns.iloc[i]
            rank = np.sum(window_returns <= current_return)
            percent_rank[i] = (rank / len(window_returns)) * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    return np.nan_to_num(crsi, nan=50.0)

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    choppiness = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr_vals[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16_4h = calculate_hma(close, 16)
    hma_48_4h = calculate_hma(close, 48)
    rsi_14_4h = calculate_rsi(close, period=14)
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_14_4h = calculate_atr(high, low, close, period=14)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(high, low, period=20)
    
    # Calculate 12h HMA for intermediate trend (aligned properly)
    hma_21_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h_raw)
    
    # Calculate 1d HMA for macro bias (aligned properly)
    hma_21_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14_4h[i]) or atr_14_4h[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_16_4h[i]) or np.isnan(hma_48_4h[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_21_12h_aligned[i]) or np.isnan(hma_21_1d_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_21_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_21_1d_aligned[i]
        macro_bullish = price_above_hma_1d
        macro_bearish = price_below_hma_1d
        
        # === INTERMEDIATE TREND (12h HMA) ===
        price_above_hma_12h = close[i] > hma_21_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_21_12h_aligned[i]
        trend_12h_bullish = price_above_hma_12h
        trend_12h_bearish = price_below_hma_12h
        
        # === 4h TREND (HMA crossover) ===
        hma_bullish_4h = hma_16_4h[i] > hma_48_4h[i]
        hma_bearish_4h = hma_16_4h[i] < hma_48_4h[i]
        
        # === CHOPPINESS REGIME ===
        choppy_market = chop_4h[i] > 55.0  # range/mean reversion regime
        trending_market = chop_4h[i] < 45.0  # trend following regime
        
        # === CONNORS RSI EXTREMES (mean reversion signals) ===
        crsi_oversold = crsi_4h[i] < 15.0  # strong buy signal
        crsi_overbought = crsi_4h[i] > 85.0  # strong sell signal
        crsi_neutral = 30.0 <= crsi_4h[i] <= 70.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper_4h[i] * 0.995
        donchian_breakout_short = close[i] < donchian_lower_4h[i] * 1.005
        
        # === DETERMINE DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY CONDITIONS (multiple paths to ensure trades)
        long_score = 0
        
        # Path 1: CRSI oversold + macro bullish (mean reversion in uptrend)
        if crsi_oversold and macro_bullish:
            long_score += 2
        
        # Path 2: CRSI oversold + 12h trend bullish
        if crsi_oversold and trend_12h_bullish:
            long_score += 2
        
        # Path 3: HMA bullish + RSI pullback (trend continuation)
        if hma_bullish_4h and 35.0 <= rsi_14_4h[i] <= 55.0:
            long_score += 1
        
        # Path 4: Donchian breakout + trend alignment
        if donchian_breakout_long and (macro_bullish or trend_12h_bullish):
            long_score += 1
        
        # Path 5: Choppy market + CRSI oversold (pure mean reversion)
        if choppy_market and crsi_oversold:
            long_score += 1
        
        # SHORT ENTRY CONDITIONS
        short_score = 0
        
        # Path 1: CRSI overbought + macro bearish
        if crsi_overbought and macro_bearish:
            short_score += 2
        
        # Path 2: CRSI overbought + 12h trend bearish
        if crsi_overbought and trend_12h_bearish:
            short_score += 2
        
        # Path 3: HMA bearish + RSI pullback
        if hma_bearish_4h and 45.0 <= rsi_14_4h[i] <= 65.0:
            short_score += 1
        
        # Path 4: Donchian breakdown + trend alignment
        if donchian_breakout_short and (macro_bearish or trend_12h_bearish):
            short_score += 1
        
        # Path 5: Choppy market + CRSI overbought
        if choppy_market and crsi_overbought:
            short_score += 1
        
        # Enter long if score >= 2 (at least 2 confluence factors)
        if long_score >= 2:
            if macro_bullish and trend_12h_bullish:
                desired_signal = POSITION_SIZE_FULL
            else:
                desired_signal = POSITION_SIZE_HALF
        
        # Enter short if score >= 2
        elif short_score >= 2:
            if macro_bearish and trend_12h_bearish:
                desired_signal = -POSITION_SIZE_FULL
            else:
                desired_signal = -POSITION_SIZE_HALF
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14_4h[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14_4h[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_bearish_4h and rsi_14_4h[i] > 60.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish_4h and rsi_14_4h[i] < 40.0:
            desired_signal = 0.0
        
        # === MACRO REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bearish and trend_12h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bullish and trend_12h_bullish:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit on mean reversion) ===
        if in_position and position_side > 0 and crsi_4h[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_4h[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if trend still valid ===
        if in_position and desired_signal == 0.0:
            if position_side > 0 and hma_bullish_4h and rsi_14_4h[i] < 80.0 and crsi_4h[i] < 80.0:
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and hma_bearish_4h and rsi_14_4h[i] > 20.0 and crsi_4h[i] > 20.0:
                desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals