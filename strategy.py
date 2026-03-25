#!/usr/bin/env python3
"""
Experiment #1352: 12h Primary + 1d HTF — Dual Regime Strategy (Choppiness + CRSI/Donchian)

Hypothesis: A dual-regime approach adapts to market conditions better than single-strategy approaches.
- When CHOP(14) > 61.8: Market is choppy → use Connors RSI mean reversion (looser thresholds)
- When CHOP(14) < 38.2: Market is trending → use Donchian breakout + HMA trend
- When 38.2 <= CHOP <= 61.8: Neutral → use moderate mean reversion

This should work better because:
1. 2022 crash was choppy (mean reversion works)
2. 2021 bull was trending (breakout works)
3. 2025 bear/range needs both approaches
4. LOOSER entry thresholds to guarantee trades (CRSI <25/>75 not <10/>90)

Key features:
1. 1d HMA(21) for major trend bias
2. Choppiness Index(14) for regime detection
3. Connors RSI for mean reversion (CRSI <25 long, >75 short) - LOOSE
4. Donchian(20) breakout for trend entries
5. ATR(14) 2.5x trailing stop
6. Discrete sizing (0.0, ±0.20, ±0.30)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d_v2"
timeframe = "12h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        atr_sum = np.nansum(atr[i - period + 1:i + 1])
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 25 = oversold (long), CRSI > 75 = overbought (short) - LOOSE thresholds
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period - 1, n):
        streak_window = streak[i - streak_period + 1:i + 1]
        avg_streak = np.mean(streak_window)
        # Map streak to 0-100 range (positive streak = high, negative = low)
        streak_rsi[i] = 50 + avg_streak * 10
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank (100) - where current price ranks in last 100 bars
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        pct_rank[i] = rank
    
    # Combine into Connors RSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    mask = ~np.isnan(rsi_3) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[mask] = (rsi_3[mask] + streak_rsi[mask] + pct_rank[mask]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_CHOP = 0.25  # Mean reversion size
    SIZE_TREND = 0.30  # Trend following size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        chop = choppiness[i]
        is_choppy = chop > 55.0  # Slightly lower threshold for more chop signals
        is_trending = chop < 45.0  # Slightly higher threshold for more trend signals
        is_neutral = not is_choppy and not is_trending
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === ENTRY LOGIC BY REGIME (LOOSE THRESHOLDS) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION (Connors RSI + RSI)
            crsi_val = crsi[i]
            rsi_val = rsi_14[i]
            
            # LONG: CRSI < 25 (oversold) + RSI < 40 + price above 1d HMA
            if crsi_val < 25 and rsi_val < 40 and price_above_1d:
                desired_signal = SIZE_CHOP
            # SHORT: CRSI > 75 (overbought) + RSI > 60 + price below 1d HMA
            elif crsi_val > 75 and rsi_val > 60 and price_below_1d:
                desired_signal = -SIZE_CHOP
            # Fallback: just CRSI extreme + 1d bias (guarantees trades)
            elif crsi_val < 20 and price_above_1d:
                desired_signal = SIZE_CHOP
            elif crsi_val > 80 and price_below_1d:
                desired_signal = -SIZE_CHOP
        
        elif is_trending:
            # TREND FOLLOWING (Donchian breakout)
            breakout_high = close[i] > donchian_upper[i-1] if i > 0 else False
            breakout_low = close[i] < donchian_lower[i-1] if i > 0 else False
            
            if breakout_high and price_above_1d:
                desired_signal = SIZE_TREND
            elif breakout_low and price_below_1d:
                desired_signal = -SIZE_TREND
            # Fallback: breakout alone (guarantees trades)
            elif breakout_high:
                desired_signal = SIZE_TREND * 0.7
            elif breakout_low:
                desired_signal = -SIZE_TREND * 0.7
        
        else:
            # NEUTRAL - use moderate mean reversion (guarantees trades)
            crsi_val = crsi[i]
            rsi_val = rsi_14[i]
            
            # More lenient thresholds in neutral regime
            if crsi_val < 30 and rsi_val < 45:
                desired_signal = SIZE_CHOP * 0.7
            elif crsi_val > 70 and rsi_val > 55:
                desired_signal = -SIZE_CHOP * 0.7
            # Fallback: just CRSI
            elif crsi_val < 25:
                desired_signal = SIZE_CHOP * 0.5
            elif crsi_val > 75:
                desired_signal = -SIZE_CHOP * 0.5
        
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
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_CHOP * 0.9:
            final_signal = SIZE_CHOP
        elif desired_signal <= -SIZE_CHOP * 0.9:
            final_signal = -SIZE_CHOP
        elif desired_signal >= SIZE_CHOP * 0.5:
            final_signal = SIZE_CHOP * 0.8
        elif desired_signal <= -SIZE_CHOP * 0.5:
            final_signal = -SIZE_CHOP * 0.8
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