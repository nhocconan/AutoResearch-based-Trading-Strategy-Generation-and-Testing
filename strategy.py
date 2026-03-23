#!/usr/bin/env python3
"""
Experiment #111: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + Donchian

Hypothesis: Previous strategies failed due to overly complex regime detection and too many
conflicting filters. This strategy simplifies by using:

1) 1w HMA(21) for macro trend bias — only trade in direction of weekly trend
2) 1d Choppiness Index(14) for regime detection — CHOP>61.8=range (mean revert), CHOP<38.2=trend
3) 4h Connors RSI for entry timing — proven 75% win rate on reversals
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4) Donchian(20) for exit — opposite break closes position
5) ATR(14) trailing stop at 2.5x — locks profits, limits drawdown

Regime-Adaptive Logic:
- RANGE (CHOP>61.8): Mean reversion — long CRSI<15, short CRSI>85
- TREND (CHOP<38.2): Trend follow — long if price>1w_HMA, short if price<1w_HMA
- NEUTRAL (38.2-61.8): Only trade with 1w HMA bias + CRSI extreme

Why this should work:
- Connors RSI proven on 4h timeframe (75% win rate on reversals)
- Choppiness Index filters false breakouts in ranging markets
- 1w HMA prevents counter-trend trades in bear markets (2022 crash)
- 4h naturally produces 25-40 trades/year (low fee drag)
- Simpler logic = more robust across BTC/ETH/SOL

Position size: 0.25 base, 0.30 with strong confluence
Stoploss: 2.5*ATR trailing
Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_donchian_1d1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_streak_rsi(close, period=2):
    """Calculate RSI of up/down streak length (Connors RSI component)."""
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (long streaks = extreme values)
    abs_streak = np.abs(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(period, len(close)):
        if np.max(abs_streak[i-period+1:i+1]) == 0:
            streak_rsi[i] = 50.0
        else:
            # Normalize streak to 0-100 scale
            streak_rsi[i] = min(100, max(0, 50 + streak[i] * 10))
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank (Connors RSI component) — where current price ranks vs last N bars."""
    pr = np.zeros(len(close))
    for i in range(period, len(close)):
        window = close[i-period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        pr[i] = (count_below / (period - 1)) * 100.0
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3."""
    rsi_3 = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    crsi = (rsi_3 + streak_rsi + pr) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    choppiness = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d Choppiness Index for regime detection
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    crsi_4h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(crsi_4h[i]) or np.isnan(sma_200[i]):
            continue
        
        # === REGIME DETECTION (1d Choppiness) ===
        chop = chop_1d_aligned[i]
        is_ranging = chop > 61.8
        is_trending = chop < 38.2
        is_neutral = not is_ranging and not is_trending
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        price_above_sma_200 = close[i] > sma_200[i]
        price_below_sma_200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi = crsi_4h[i]
        crsi_oversold = crsi < 15.0
        crsi_overbought = crsi > 85.0
        crsi_extreme_long = crsi < 20.0
        crsi_extreme_short = crsi > 80.0
        
        # === DONCHIAN BREAKOUT ===
        prev_high = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_low = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        breakout_long = close[i] > prev_high
        breakout_short = close[i] < prev_low
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        new_signal = 0.0
        entry_strength = 1  # 1=base, 2=max
        
        if is_ranging:
            # RANGE REGIME: Mean reversion at extremes
            if crsi_oversold and price_above_sma_200:
                new_signal = POSITION_SIZE_BASE
                if crsi < 10.0:
                    new_signal = POSITION_SIZE_MAX
                    entry_strength = 2
            elif crsi_overbought and price_below_sma_200:
                new_signal = -POSITION_SIZE_BASE
                if crsi > 90.0:
                    new_signal = -POSITION_SIZE_MAX
                    entry_strength = 2
        
        elif is_trending:
            # TREND REGIME: Follow 1w HMA direction with CRSI pullback entry
            if price_above_hma_1w and crsi_extreme_long:
                new_signal = POSITION_SIZE_BASE
                if breakout_long:
                    new_signal = POSITION_SIZE_MAX
                    entry_strength = 2
            elif price_below_hma_1w and crsi_extreme_short:
                new_signal = -POSITION_SIZE_BASE
                if breakout_short:
                    new_signal = -POSITION_SIZE_MAX
                    entry_strength = 2
        
        else:
            # NEUTRAL REGIME: Only trade with 1w bias + strong CRSI
            if price_above_hma_1w and crsi_extreme_long:
                new_signal = POSITION_SIZE_BASE
            elif price_below_hma_1w and crsi_extreme_short:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if position intact and no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if above Donchian mid and CRSI not overbought
                if close[i] > donchian_mid[i] and crsi < 80.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if below Donchian mid and CRSI not oversold
                if close[i] < donchian_mid[i] and crsi > 20.0:
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
        
        # === EXIT ON OPPOSITE DONCHIAN BREAK ===
        if in_position and position_side > 0 and breakout_short:
            new_signal = 0.0
        
        if in_position and position_side < 0 and breakout_long:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL (1w HMA cross) ===
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1w:
            new_signal = 0.0
        
        # === TAKE PROFIT ON CRSI EXTREME ===
        if in_position and position_side > 0 and crsi > 85.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi < 15.0:
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