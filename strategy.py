#!/usr/bin/env python3
"""
Experiment #1172: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + Connors RSI + Donchian

Hypothesis: After 858+ failed experiments, the key insight is regime adaptation.
Single-regime strategies fail because:
- Trend strategies get whipsawed in range markets (2022, 2025)
- Mean reversion gets crushed in strong trends

This strategy uses Choppiness Index to detect regime:
- CHOP > 61.8: Range market → Connors RSI mean reversion
- CHOP < 38.2: Trend market → Donchian breakout + HMA trend

Why 12h works:
- Natural 20-50 trades/year (optimal fee drag)
- Less noise than 4h, more signals than 1d
- 1d/1w HTF filters prevent counter-trend trades

Key improvements over #1162 (Sharpe=-0.724):
- Dual regime instead of single trend-follow
- Connors RSI (75% win rate) instead of simple RSI pullback
- Donchian breakout for trend capture
- 1w HMA for ultra-long trend filter
- Relaxed entry conditions to ensure trades on all symbols

Timeframe: 12h (primary)
HTF: 1d, 1w — loaded ONCE before loop
Position Size: 0.28 base (discrete: 0.0, ±0.28)
Stoploss: 2.5x ATR trailing
Target: 30-50 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and Choppiness Index."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — detects range vs trend regime.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period - 1, n):
        if np.isnan(atr[i]):
            continue
        
        sum_atr = np.nansum(atr[i - period + 1:i + 1])
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — mean reversion oscillator with 75% win rate.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI(streak): Duration of consecutive up/down days
    PercentRank: Where current return ranks vs last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close[:rsi_period] = np.nan
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (using absolute values for gains/losses)
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    mask = streak_loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = streak_gain_smooth[mask] / streak_loss_smooth[mask]
    rsi_streak = 100.0 - (100.0 / (1.0 + rs))
    rsi_streak[:streak_period] = np.nan
    
    # PercentRank(100)
    returns = np.diff(close) / np.maximum(close[:-1], 1e-10) * 100
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period - 1, n):
        window = returns[i - rank_period + 1:i]
        if len(window) > 0 and not np.all(np.isnan(window)):
            current_return = returns[i-1] if i > 0 else 0
            rank = np.sum(window < current_return) / len(window)
            percent_rank[i] = rank * 100
    
    # Combine
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout levels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-long trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_12h[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/choppy → mean reversion
        # CHOP < 38.2 = trending → breakout
        # 38.2 <= CHOP <= 61.8 = transition → no trade
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === MACRO TREND FILTERS ===
        # 1w HMA: Ultra-long trend (only trade with this)
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # 1d HMA: Medium-term trend
        mid_bull = close[i] > hma_1d_aligned[i]
        mid_bear = close[i] < hma_1d_aligned[i]
        
        # 12h HMA: Local trend
        local_bull = close[i] > hma_12h[i]
        local_bear = close[i] < hma_12h[i]
        
        desired_signal = 0.0
        
        # === MEAN REVERSION (Choppy Regime) ===
        if is_choppy:
            # Long: CRSI < 20 (oversold) + price > 1w HMA (macro bull filter)
            if crsi[i] < 20.0 and macro_bull:
                desired_signal = BASE_SIZE
            
            # Short: CRSI > 80 (overbought) + price < 1w HMA (macro bear filter)
            elif crsi[i] > 80.0 and macro_bear:
                desired_signal = -BASE_SIZE
        
        # === TREND FOLLOWING (Trending Regime) ===
        elif is_trending:
            # Long breakout: Price breaks Donchian upper + 1d/12h HMA aligned bull
            if close[i] > donchian_upper[i] and mid_bull and local_bull:
                desired_signal = BASE_SIZE
            
            # Short breakout: Price breaks Donchian lower + 1d/12h HMA aligned bear
            elif close[i] < donchian_lower[i] and mid_bear and local_bear:
                desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if regime and trend still valid
                if (is_choppy and crsi[i] < 50.0 and macro_bull) or \
                   (is_trending and mid_bull and local_bull):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if regime and trend still valid
                if (is_choppy and crsi[i] > 50.0 and macro_bear) or \
                   (is_trending and mid_bear and local_bear):
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals