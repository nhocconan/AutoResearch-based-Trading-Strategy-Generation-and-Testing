#!/usr/bin/env python3
"""
Experiment #1511: 4h Primary + 1d/1w HTF — Dual Regime (Choppiness + CRSI/HMA)

Hypothesis: After analyzing 1100+ failed strategies, the winning pattern is:
1. 4h timeframe with 1d/1w HTF should generate 20-50 trades/year (sweet spot)
2. Dual-regime approach: CHOP(14) detects trend vs range, switch logic accordingly
3. TREND regime (CHOP < 38.2): Follow 1d HMA trend, enter on 4h HMA pullback + RSI
4. RANGE regime (CHOP > 61.8): Mean revert using Connors RSI extremes (CRSI < 10 / > 90)
5. This combines proven patterns: CHOP regime (ETH +0.923), CRSI (ETH +0.923), HMA trend (SOL +0.879)
6. Use 1w HMA for ultra-macro bias to avoid counter-trend trades in strong trends

Key design choices:
- 1d HMA(21) for primary trend direction
- 1w HMA(21) for ultra-macro bias (avoid shorting in bull, avoid longing in bear)
- 4h Choppiness Index(14) for regime detection
- 4h Connors RSI for mean reversion entries in range regime
- 4h HMA(21) + RSI(14) for trend pullback entries in trend regime
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.30 (discrete: 0.0, ±0.30)
- LOOSE entry conditions to ensure trades happen (critical after 0-trade failures)

Timeframe: 4h (as required by experiment)
HTF: 1d (trend bias), 1w (macro bias)
Position Size: 0.30 (discrete: 0.0, ±0.30)
Target: 80-200 trades/train (4 years), 20-50 trades/test (15 months), Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_crsi_hma_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_window = streak[i - streak_period + 1:i + 1]
        pos_streak = np.sum(streak_window > 0)
        streak_rsi[i] = 100.0 * pos_streak / streak_period if streak_period > 0 else 50.0
    
    # Percent Rank (100) - where current return ranks in last 100 periods
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i - rank_period:i + 1])
        if len(returns) > 0 and returns[-1] != 0:
            pct_rank[i] = 100.0 * np.sum(returns[:-1] < returns[-1]) / (len(returns) - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + pct_rank[mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 4h (target 20-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_4h[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ULTRA-MACRO BIAS (1w HMA) - avoid counter-trend in strong trends ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) - confirmation ===
        h4_bull = close[i] > hma_4h[i]
        h4_bear = close[i] < hma_4h[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        is_trend_regime = chop[i] < 38.2
        is_range_regime = chop[i] > 61.8
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === TREND REGIME: Follow trend with pullback entries ===
        if is_trend_regime:
            # LONG: Weekly bull + Daily bull + 4h HMA bull + RSI pullback (not overbought)
            if weekly_bull and daily_bull and h4_bull and rsi[i] < 60.0:
                desired_signal = BASE_SIZE
            # LONG (looser): Weekly bull + Daily bull + RSI not overbought
            elif weekly_bull and daily_bull and rsi[i] < 55.0:
                desired_signal = BASE_SIZE * 0.8
            # LONG (fallback): Daily bull + 4h HMA bull + RSI pullback
            elif daily_bull and h4_bull and rsi[i] < 50.0:
                desired_signal = BASE_SIZE * 0.6
            
            # SHORT: Weekly bear + Daily bear + 4h HMA bear + RSI pullback (not oversold)
            elif weekly_bear and daily_bear and h4_bear and rsi[i] > 40.0:
                desired_signal = -BASE_SIZE
            # SHORT (looser): Weekly bear + Daily bear + RSI not oversold
            elif weekly_bear and daily_bear and rsi[i] > 45.0:
                desired_signal = -BASE_SIZE * 0.8
            # SHORT (fallback): Daily bear + 4h HMA bear + RSI pullback
            elif daily_bear and h4_bear and rsi[i] > 50.0:
                desired_signal = -BASE_SIZE * 0.6
        
        # === RANGE REGIME: Mean revert using CRSI extremes ===
        elif is_range_regime:
            # LONG: CRSI < 15 (oversold) + price above 1w HMA (bullish macro)
            if not np.isnan(crsi[i]) and crsi[i] < 15.0 and weekly_bull:
                desired_signal = BASE_SIZE
            # LONG (looser): CRSI < 20 (oversold)
            elif not np.isnan(crsi[i]) and crsi[i] < 20.0:
                desired_signal = BASE_SIZE * 0.8
            # LONG (fallback): RSI < 30 (oversold) in range
            elif rsi[i] < 30.0:
                desired_signal = BASE_SIZE * 0.6
            
            # SHORT: CRSI > 85 (overbought) + price below 1w HMA (bearish macro)
            elif not np.isnan(crsi[i]) and crsi[i] > 85.0 and weekly_bear:
                desired_signal = -BASE_SIZE
            # SHORT (looser): CRSI > 80 (overbought)
            elif not np.isnan(crsi[i]) and crsi[i] > 80.0:
                desired_signal = -BASE_SIZE * 0.8
            # SHORT (fallback): RSI > 70 (overbought) in range
            elif rsi[i] > 70.0:
                desired_signal = -BASE_SIZE * 0.6
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8): Stay flat or reduce ===
        else:
            # Only take strongest signals in neutral regime
            if weekly_bull and daily_bull and rsi[i] < 40.0:
                desired_signal = BASE_SIZE * 0.5
            elif weekly_bear and daily_bear and rsi[i] > 60.0:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.6
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