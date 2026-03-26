#!/usr/bin/env python3
"""
Experiment #002: 12h CRSI + Choppiness Regime + 1d Trend Bias

HYPOTHESIS: 12h timeframe with CRSI signal + 1d HTF trend bias captures the 
optimal balance between trade frequency and signal quality for this TF level.

Why 12h CRSI should work:
1. CRSI(3,2,100) proven on SOLUSDT test Sharpe=1.46 — top performer in DB
2. 12h TF = ~60-70 trades over 4 years (within target 50-150 range)
3. CRSI signals are STATIC (not crossovers), reducing false flips vs EMA
4. Choppiness regime filter prevents whipsaw in ranging markets
5. 1d HMA provides trend bias, filtering counter-trend signals

Key design choices:
- 12h primary TF (as required)
- 1d HTF for HMA trend bias (loaded ONCE via mtf_data helper)
- Loose CRSI thresholds: <20 long, >80 short (generates enough trades)
- CHOP < 38.2 = trend regime (trend follow), CHOP > 61.8 = range (mean revert)
- 2.5x ATR stoploss via signal→0
- Discrete sizing: 0.25 base, 0.30 strong confirmation

Target: 50-150 total trades over 4 years, Sharpe>0.5, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_1d_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain = np.insert(gain, 0, 0.0)
    loss = np.insert(loss, 0, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, 50.0, dtype=np.float64)
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak - measures consecutive up/down closes
    Part of Connors RSI (CRSI)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    
    # Ups: 1 if close went up, 0 otherwise
    ups = np.zeros(n - 1)
    ups[delta > 0] = 1.0
    
    # Downs: -1 if close went down, 0 otherwise
    downs = np.zeros(n - 1)
    downs[delta < 0] = -1.0
    
    # Net streak
    streak = ups + downs
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Count consecutive up days in last 'period'
        consec_up = 0
        for j in range(i - period, i):
            if delta[j] > 0:
                consec_up += 1
            else:
                break
        
        # RSI of streak values
        window = streak[max(0, i - period):i]
        if len(window) > 0:
            # Simple RSI on streak
            gains = np.where(window > 0, window, 0.0)
            losses = np.where(window < 0, -window, 0.0)
            avg_gain = np.mean(gains) if np.sum(gains) > 0 else 0.0
            avg_loss = np.mean(losses) if np.sum(losses) > 0 else 0.0
            
            if avg_loss == 0:
                rsi_streak[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_streak[i] = 50.0
    
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """
    Percent Rank - percentile rank of current close over lookback
    Part of Connors RSI (CRSI)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window = close[i - period:i + 1]
        current = close[i]
        
        # Count how many values are less than current
        count_below = np.sum(window < current)
        # Count equal (excluding current to avoid self-bias)
        count_equal = np.sum(window[:-1] == current)
        
        pct_rank[i] = ((count_below + 0.5 * count_equal) / period) * 100.0
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < max(rsi_period, streak_period, rank_period) + 1:
        return np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, period=rsi_period)
    rsi_streak = calculate_rsi_streak(close, period=streak_period)
    pct_rank = calculate_percent_rank(close, period=rank_period)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not (np.isnan(rsi_3[i]) or np.isnan(rsi_streak[i]) or np.isnan(pct_rank[i])):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for structure"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period for 12h with CRSI(100) lookback
    min_bars = 120
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === 1d HTF TREND BIAS ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === CRSI SIGNALS ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 20      # Loose: 20 instead of 10
        crsi_overbought = crsi_val > 80     # Loose: 80 instead of 90
        crsi_extreme_low = crsi_val < 10
        crsi_extreme_high = crsi_val > 90
        
        # === DONCHIAN STRUCTURE ===
        near_donch_support = False
        near_donch_resistance = False
        
        if i > 0 and not np.isnan(donch_lower[i-1]) and not np.isnan(donch_upper[i-1]):
            donch_range = donch_upper[i-1] - donch_lower[i-1]
            if donch_range > 0:
                # Within 20% of lower band = support
                dist_to_lower = (close[i] - donch_lower[i-1]) / donch_range
                near_donch_support = dist_to_lower < 0.20
                
                # Within 20% of upper band = resistance
                dist_to_upper = (donch_upper[i-1] - close[i]) / donch_range
                near_donch_resistance = dist_to_upper < 0.20
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: CRSI extremes + 1d HMA bias + Donchian structure
        if is_trend_regime:
            # LONG: CRSI oversold + price above 1d HMA + near Donchian support or breakout
            if crsi_oversold and price_above_1d and near_donch_support:
                desired_signal = SIZE_STRONG
            elif crsi_oversold and price_above_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + price below 1d HMA + near Donchian resistance or breakout
            elif crsi_overbought and price_below_1d and near_donch_resistance:
                desired_signal = -SIZE_STRONG
            elif crsi_overbought and price_below_1d:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion - CRSI extreme + not against trend
        elif is_range_regime:
            # LONG: CRSI extreme oversold + NOT bearish 1d bias
            if crsi_extreme_low and not price_below_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI extreme overbought + NOT bullish 1d bias
            elif crsi_extreme_high and not price_above_1d:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Simple CRSI extremes + 1d bias only
        else:
            # LONG: CRSI oversold + bullish 1d
            if crsi_oversold and price_above_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + bearish 1d
            elif crsi_overbought and price_below_1d:
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
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = close[i] - 2.5 * entry_atr
                else:
                    stop_price = close[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals