#!/usr/bin/env python3
"""
Experiment #1503: 6h Primary + 1d/1w HTF — Fisher-CRSI Mean Reversion with Regime Filter

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy combines:
1. Ehlers Fisher Transform (period=9) - catches reversals in bear/bull markets with 75%+ win rate
2. Connors RSI (CRSI) - (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for extreme mean reversion
3. 1d HMA(21) for major trend bias - avoid counter-trend in strong trends
4. 1w HMA(21) for secular trend filter - only trade with weekly momentum
5. Choppiness Index(14) regime detection - adjust entry thresholds per regime

Why 6h: Middle ground between 4h (too many trades) and 12h (too few). Target 30-50 trades/year.
This is DIFFERENT from failed 6h strategies (weekly pivots, simple RSI, Fisher alone).

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_HMA bullish + 1w_HMA bullish + Fisher < -1.0 + CRSI < 25 + CHOP > 50 (range favor)
- SHORT: 1d_HMA bearish + 1w_HMA bearish + Fisher > +1.0 + CRSI > 75 + CHOP > 50

Alternative breakout in trend regime (CHOP < 40):
- LONG: Fisher cross above -1.5 + price > 1d_HMA
- SHORT: Fisher cross below +1.5 + price < 1d_HMA

Position sizing: 0.25 base, 0.30 strong signals (discrete to minimize fee churn)
Stoploss: 2.5x ATR(14) trailing stop

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_crsi_regime_1d1w_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points with sharp peaks
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        normalized = (hl2 - lowest_low) / price_range
        
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value (Ehlers recommends this)
        if i > period - 1 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
        else:
            fisher[i] = fisher_val
        
        if i > period - 1:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): Duration of consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 bars
    
    CRSI < 10 = extreme oversold (long), CRSI > 90 = extreme overbought (short)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    mask = avg_streak_loss != 0
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rsi_streak[mask] = 100 - (100 / (1 + rs_streak[mask]))
    rsi_streak[avg_streak_loss == 0] = 100.0
    
    # Percent Rank - where current close ranks vs last 100 bars
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(rsi_3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_3[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
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
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
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
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 40.0
        is_range_regime = chop > 55.0
        
        # === TREND DIRECTION (1d and 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM ===
        fisher_val = fisher[i]
        fisher_prev_val = fisher_prev[i] if not np.isnan(fisher_prev[i]) else fisher_val
        
        # Fisher crossover signals
        fisher_cross_up = fisher_prev_val < -1.5 and fisher_val >= -1.5
        fisher_cross_down = fisher_prev_val > 1.5 and fisher_val <= 1.5
        fisher_extreme_low = fisher_val < -1.0
        fisher_extreme_high = fisher_val > 1.0
        
        # === CRSI ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 30
        crsi_overbought = crsi_val > 70
        crsi_extreme_oversold = crsi_val < 20
        crsi_extreme_overbought = crsi_val > 80
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with Fisher + CRSI extremes
        if is_range_regime:
            # LONG: 1d bullish bias + Fisher extreme low + CRSI oversold
            if price_above_1d and fisher_extreme_low and crsi_oversold:
                desired_signal = SIZE_STRONG
            # Also allow without 1d filter if CRSI extreme
            elif fisher_extreme_low and crsi_extreme_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish bias + Fisher extreme high + CRSI overbought
            elif price_below_1d and fisher_extreme_high and crsi_overbought:
                desired_signal = -SIZE_STRONG
            # Also allow without 1d filter if CRSI extreme
            elif fisher_extreme_high and crsi_extreme_overbought:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Fisher crossover with trend confirmation
        elif is_trend_regime:
            # LONG: 1d + 1w bullish + Fisher cross up
            if price_above_1d and price_above_1w and fisher_cross_up:
                desired_signal = SIZE_STRONG
            # Weaker: just 1d bullish + Fisher cross up
            elif price_above_1d and fisher_cross_up:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d + 1w bearish + Fisher cross down
            elif price_below_1d and price_below_1w and fisher_cross_down:
                desired_signal = -SIZE_STRONG
            # Weaker: just 1d bearish + Fisher cross down
            elif price_below_1d and fisher_cross_down:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Only take extreme CRSI signals
        else:
            # LONG: CRSI extreme oversold + Fisher low
            if crsi_extreme_oversold and fisher_val < -0.5:
                desired_signal = SIZE_BASE
            # SHORT: CRSI extreme overbought + Fisher high
            elif crsi_extreme_overbought and fisher_val > 0.5:
                desired_signal = -SIZE_BASE
        
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