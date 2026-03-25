#!/usr/bin/env python3
"""
Experiment #1643: 6h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime

Hypothesis: 6h timeframe fills the gap between 4h (too many trades) and 12h (too few).
Connors RSI (CRSI) has proven 75% win rate in bear/range markets (2022-2024).
Combined with Choppiness Index regime filter and 1d/1w HMA trend bias.

Key improvements over #1640 (mtf_6h_fisher_rsi_regime_1d1w_loose_v1, Sharpe=-0.464):
1. CRSI instead of Fisher - better for mean reversion in bear markets
2. LOOSER CHOP thresholds (50/55 instead of 38.2/61.8) to guarantee more trades
3. Simpler 1w HMA bias (just direction, not complex regime)
4. Volatility filter (ATR ratio) to avoid low-vol whipsaws
5. Discrete signals: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry logic (LOOSE to guarantee ≥30 trades/train):
- TREND (CHOP<50): CRSI pullback + 1d/1w HMA alignment + ATR confirmation
- RANGE (CHOP>55): CRSI extremes (<15 or >85) + Bollinger touch
- NEUTRAL: 1d HMA bias + CRSI moderate extremes (<25 or >75)

Target: Sharpe>0.6, trades≥30 train, trades≥5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_regime_1d1w_loose_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    Streak RSI component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        streak_window = streak[i - period + 1:i + 1]
        up_streaks = np.sum(streak_window > 0)
        down_streaks = np.sum(streak_window < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures current price change vs past period changes
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        current_change = close[i] - close[i-1] if i > 0 else 0
        past_changes = np.diff(close[i - period:i + 1])
        
        if len(past_changes) > 0:
            count_lower = np.sum(past_changes < current_change)
            pct_rank[i] = 100.0 * count_lower / len(past_changes)
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion in bear markets
    """
    rsi_fast = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_streak_rsi(close, period=streak_period)
    pct_rank = calculate_percent_rank(close, period=rank_period)
    
    crsi = np.full(len(close), np.nan, dtype=np.float64)
    for i in range(len(close)):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    Using LOOSER thresholds (50/55) to guarantee more trades
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_atr_ratio(atr_short, atr_long):
    """ATR ratio for volatility filter"""
    ratio = np.full(len(atr_short), np.nan, dtype=np.float64)
    mask = (atr_long > 1e-10) & (~np.isnan(atr_short)) & (~np.isnan(atr_long))
    ratio[mask] = atr_short[mask] / atr_long[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = calculate_atr_ratio(atr_7, atr_30)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 120  # Need enough bars for CRSI percent_rank(100)
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]):
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
        
        # === REGIME DETECTION (Choppiness Index - LOOSE thresholds) ===
        chop = chop_14[i]
        is_trend_regime = chop < 50.0  # LOOSER than 38.2
        is_range_regime = chop > 55.0  # LOOSER than 61.8
        
        # === TREND DIRECTION (1d and 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY FILTER ===
        vol_spike = atr_ratio[i] > 1.8  # ATR(7) > 1.8x ATR(30) = vol spike
        vol_normal = atr_ratio[i] < 1.3  # Normal volatility
        
        # === CRSI SIGNALS (LOOSE thresholds for trades) ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 25.0  # LOOSE (was 15)
        crsi_overbought = crsi_val > 75.0  # LOOSE (was 85)
        crsi_extreme_low = crsi_val < 15.0
        crsi_extreme_high = crsi_val > 85.0
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.01
        bb_touch_upper = close[i] >= bb_upper[i] * 0.99
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: CRSI pullback + HTF alignment
        if is_trend_regime:
            # LONG: 1d+1w bullish + CRSI pullback (not oversold, just dipped)
            if price_above_1d and price_above_1w and crsi_val < 50 and crsi_val > 20:
                desired_signal = SIZE_STRONG if vol_normal else SIZE_BASE
            
            # SHORT: 1d+1w bearish + CRSI rally (not overbought, just rose)
            elif price_below_1d and price_below_1w and crsi_val > 50 and crsi_val < 80:
                desired_signal = -SIZE_STRONG if vol_normal else -SIZE_BASE
        
        # RANGE REGIME: CRSI extremes + Bollinger touch (mean reversion)
        elif is_range_regime:
            # LONG: CRSI extreme low + price at BB lower
            if crsi_extreme_low and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI extreme high + price at BB upper
            elif crsi_extreme_high and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: 1d HMA bias + CRSI moderate extremes
        else:
            # LONG: 1d bullish + CRSI oversold
            if price_above_1d and crsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + CRSI overbought
            elif price_below_1d and crsi_overbought:
                desired_signal = -SIZE_BASE
        
        # VOL SPIKE REVERSION: Additional entry on panic
        if vol_spike and crsi_extreme_low and price_above_1d:
            desired_signal = SIZE_BASE  # Buy the panic dip
        
        if vol_spike and crsi_extreme_high and price_below_1d:
            desired_signal = -SIZE_BASE  # Sell the panic rally
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals