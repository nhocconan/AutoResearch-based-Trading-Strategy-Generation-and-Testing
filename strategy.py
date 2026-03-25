#!/usr/bin/env python3
"""
Experiment #1423: 6h Primary + 1d/1w HTF — Fisher Transform + CHOP Regime + CRSI

Hypothesis: 6h timeframe is unexplored territory between proven 4h and failing 12h.
This strategy combines regime-adaptive logic:
1. CHOP(14) regime filter: <38.2 = trending (use Fisher), >61.8 = ranging (use CRSI)
2. Ehlers Fisher Transform(9): catches reversals better than RSI in bear markets
3. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for mean-reversion
4. 1d HMA(21) + 1w HMA(21): dual HTF trend bias (both must agree)
5. ATR(14) trailing stoploss with 2.5x multiplier
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should beat Triple HMA (Sharpe=0.575):
- Fisher Transform is proven to catch bear market reversals (research-backed)
- CHOP regime filter prevents trend strategies in ranges (major failure mode)
- CRSI provides 75% win rate mean-reversion when CHOP indicates ranging
- Dual HTF (1d+1w) prevents counter-trend trades in major moves
- 6h TF = ~35-50 trades/year (fee-efficient, not overtraded)

Entry logic (LOOSE to guarantee trades):
- TRENDING (CHOP<38.2): Fisher crosses -1.5 upward + 1d/1w HMA bullish → LONG
- TRENDING (CHOP<38.2): Fisher crosses +1.5 downward + 1d/1w HMA bearish → SHORT
- RANGING (CHOP>61.8): CRSI<15 + price>SMA200 → LONG (mean-revert long)
- RANGING (CHOP>61.8): CRSI>85 + price<SMA200 → SHORT (mean-revert short)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_crsi_regime_1d1w_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """Ehlers Fisher Transform - catches reversals in bear markets"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize price to range -1 to +1
        norm = 0.66 * ((hl2 - lowest) / (highest - lowest) - 0.5)
        norm = np.clip(norm, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + norm) / (1 - norm))
        
        # Trigger line (1-period lag of fisher)
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies trending vs ranging markets"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            atr_sum += tr
        
        # CHOP = 100 * log10(sum(ATR) / (High - Low)) / log10(period)
        range_val = highest - lowest
        if range_val > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        streak = 0
        if close[i] > close[i-1]:
            streak = 1
            j = i - 1
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            streak = -1
            j = i - 1
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            streak_rsi[i] = min(100, streak * 50)
        else:
            streak_rsi[i] = max(0, 100 + streak * 50)
    
    # Percent Rank - position of current change vs last 100 changes
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        changes = np.diff(close[i - rank_period:i + 1])
        if len(changes) > 0 and not np.any(np.isnan(changes)):
            current_change = changes[-1]
            count_below = np.sum(changes[:-1] < current_change)
            percent_rank[i] = 100.0 * count_below / (len(changes) - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, trigger = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, period=200)
    
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
    min_bars = 250  # Need enough bars for CRSI rank_period + HTF alignment
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
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
        
        # === HTF TREND BIAS (1d + 1w must agree) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # Both HTF agree bullish
        htf_bullish = price_above_1d and price_above_1w
        # Both HTF agree bearish
        htf_bearish = price_below_1d and price_below_1w
        
        # === REGIME DETECTION (CHOP) ===
        choppiness = chop[i]
        is_trending = choppiness < 38.2  # Low CHOP = trending
        is_ranging = choppiness > 61.8  # High CHOP = ranging
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Use Fisher Transform for entries
        if is_trending:
            # LONG: HTF bullish + Fisher crosses above -1.5
            if htf_bullish and trigger[i] < -1.5 and fisher[i] > -1.5:
                desired_signal = SIZE_STRONG
            
            # SHORT: HTF bearish + Fisher crosses below +1.5
            elif htf_bearish and trigger[i] > 1.5 and fisher[i] < 1.5:
                desired_signal = -SIZE_STRONG
        
        # RANGING REGIME: Use CRSI for mean-reversion
        elif is_ranging:
            # LONG: CRSI oversold + price above SMA200 (uptrend bias)
            if crsi[i] < 15 and not np.isnan(sma_200[i]) and close[i] > sma_200[i]:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + price below SMA200 (downtrend bias)
            elif crsi[i] > 85 and not np.isnan(sma_200[i]) and close[i] < sma_200[i]:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME (38.2 <= CHOP <= 61.8): Stay flat or use lighter signals
        else:
            # Optional: light mean-reversion in neutral zone
            if crsi[i] < 10:
                desired_signal = SIZE_BASE * 0.5
            elif crsi[i] > 90:
                desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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