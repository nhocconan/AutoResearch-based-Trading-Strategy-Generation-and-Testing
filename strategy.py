#!/usr/bin/env python3
"""
Experiment #1242: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) with Connors RSI

Hypothesis: 12h timeframe is proven to work (see #1236 Sharpe=0.260). Key innovation:
dual-regime approach that SWITCHES strategy based on market condition:
1. CHOPPY regime (CHOP>61.8): Mean reversion using Connors RSI extremes
2. TRENDING regime (CHOP<38.2): Trend following using Donchian breakout + HMA filter

Why this should work:
- Connors RSI has 75% win rate for mean reversion (research-backed)
- Choppiness Index is the BEST regime filter for crypto (from 200+ failed experiments)
- 12h TF naturally limits trades to 20-50/year (fee-efficient)
- 1d HMA provides macro bias, 1w HMA provides secular trend filter
- Asymmetric: only long when 1w HMA bull, only short when 1w HMA bear

Target: Sharpe > 0.612, trades >= 80 train (20/year), >= 30 test (3/year)
Timeframe: 12h (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market consolidation vs trending
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    Long: CRSI < 10 | Short: CRSI > 90
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close[:rsi_period] = np.nan
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    mask2 = streak_loss_smooth > 1e-10
    rs2 = np.zeros(n)
    rs2[mask2] = streak_gain_smooth[mask2] / streak_loss_smooth[mask2]
    rsi_streak = 100.0 - (100.0 / (1.0 + rs2))
    rsi_streak[:streak_period] = np.nan
    
    # Component 3: Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Combine all three
    valid_mask = (~np.isnan(rsi_close)) & (~np.isnan(rsi_streak)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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
    
    # Calculate and align 1w HMA for secular trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    # Donchian channels for trend breakout
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # HMA for trend confirmation on primary TF
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Slightly lower for 12h to reduce DD
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track regime for hysteresis
    prev_regime = 0  # 0=unknown, 1=trend, 2=chop
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Slightly lower threshold for more regime switches
        is_trending = chop[i] < 45.0  # Hysteresis band
        
        # Regime hysteresis
        if is_trending:
            regime = 1  # trending
        elif is_choppy:
            regime = 2  # choppy
        else:
            regime = prev_regime  # maintain previous
        prev_regime = regime
        
        # === MACRO TREND FILTERS ===
        macro_bull_1d = close[i] > hma_1d_aligned[i]
        macro_bear_1d = close[i] < hma_1d_aligned[i]
        macro_bull_1w = close[i] > hma_1w_aligned[i]
        macro_bear_1w = close[i] < hma_1w_aligned[i]
        
        # Primary trend
        trend_bull = hma_12h[i] > close[i] * 0.995  # HMA below price
        trend_bear = hma_12h[i] < close[i] * 1.005  # HMA above price
        
        # === MEAN REVERSION SIGNALS (Choppy Regime) ===
        crsi_oversold = crsi[i] < 15.0  # Long entry
        crsi_overbought = crsi[i] > 85.0  # Short entry
        
        # === TREND BREAKOUT SIGNALS (Trending Regime) ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === ENTRY LOGIC (Dual Regime) ===
        desired_signal = 0.0
        
        if regime == 2:  # CHOPPY - Mean Reversion
            # Long: CRSI oversold + price above 1d HMA (bullish bias)
            if crsi_oversold and macro_bull_1d:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + price below 1d HMA (bearish bias)
            elif crsi_overbought and macro_bear_1d:
                desired_signal = -BASE_SIZE
        
        elif regime == 1:  # TRENDING - Trend Following
            # Long: Breakout + 1w HMA bull + 1d HMA bull
            if breakout_long and macro_bull_1w and macro_bull_1d:
                desired_signal = BASE_SIZE
            # Short: Breakout + 1w HMA bear + 1d HMA bear
            elif breakout_short and macro_bear_1w and macro_bear_1d:
                desired_signal = -BASE_SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            final_signal = BASE_SIZE
        elif desired_signal < 0:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals