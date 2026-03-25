#!/usr/bin/env python3
"""
Experiment #1268: 4h Primary + 12h/1d HTF — Choppiness Regime + CRSI Entries

Hypothesis: Based on research showing Choppiness Index + Connors RSI achieved 
ETH Sharpe +0.923, this strategy uses regime-adaptive logic with LOOSE entry
conditions to guarantee sufficient trades. Key innovations:

1. CHOP(14) regime filter: <50 = trending (follow HTF), >50 = ranging (mean revert)
2. CRSI for entries: combines RSI(3) + StreakRSI + PercentRank for precise timing
3. 12h HMA(21) slope for primary trend bias (loose: just slope direction)
4. 1d HMA(21) for major regime confirmation (price above/below)
5. ATR(14) 2.5x trailing stop for risk management
6. DISCRETE sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

CRITICAL: Entry conditions are LOOSE to guarantee 20-50 trades/year on 4h:
- Trending long: 12h_HMA up + CRSI < 50 (not extreme 30)
- Trending short: 12h_HMA down + CRSI > 50 (not extreme 70)
- Ranging long: CRSI < 35 (mean reversion)
- Ranging short: CRSI > 65 (mean reversion)

Why this beats previous failures:
- Previous 4h strategies had TOO STRICT entries (0 trades)
- CRSI is more sensitive than plain RSI for entry timing
- Regime switching adapts to market conditions (bear/range/trend)
- 12h HMA slope is simpler than dual-HMA crossover (fewer conditions)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h (20-50 trades/year target)
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_hma_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan, dtype=np.float64)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
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
    """Average True Range for volatility and stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=np.float64)
    
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
        return np.full(n, np.nan, dtype=np.float64)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - identifies trending vs ranging markets
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    We use 50.0 as middle threshold for regime switching
    """
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Combines momentum, streak, and relative position for entry timing
    """
    n = len(close)
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1.0 if streak[i-1] >= 0 else 1.0
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1.0 if streak[i-1] <= 0 else -1.0
        else:
            streak[i] = 0.0
    
    # Streak RSI - use absolute streak for RSI calculation
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    
    # Percentile Rank - where current close ranks in last 100 bars
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine all three components
    start_idx = max(rsi_period + 1, streak_period + 1, rank_period)
    for i in range(start_idx, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    
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
    
    # Warmup period for indicators
    min_bars = 150
    
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
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (12h HMA slope) ===
        hma_12h_slope = 0.0
        if i >= 2 and not np.isnan(hma_12h_aligned[i-2]):
            hma_12h_slope = hma_12h_aligned[i] - hma_12h_aligned[i-2]
        
        # 1d HMA bias for major regime
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 4h price vs 4h HMA for local confirmation
        price_above_4h = close[i] > hma_4h[i]
        price_below_4h = close[i] < hma_4h[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Below 50 = trending market
        is_ranging = chop >= 50.0  # Above 50 = ranging/choppy market
        
        # === ENTRY CONDITIONS (LOOSE to ensure trades) ===
        desired_signal = 0.0
        crsi_val = crsi[i]
        
        # LONG entries
        if hma_12h_slope > 0:  # 12h trend is up
            if is_trending:
                # Trend following: enter on CRSI pullback (not too extreme)
                if crsi_val < 50.0:
                    if crsi_val < 35.0:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            else:
                # Ranging: mean reversion at lower bound
                if crsi_val < 35.0:
                    desired_signal = SIZE_BASE
        
        # SHORT entries
        elif hma_12h_slope < 0:  # 12h trend is down
            if is_trending:
                # Trend following: enter on CRSI bounce (not too extreme)
                if crsi_val > 50.0:
                    if crsi_val > 65.0:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            else:
                # Ranging: mean reversion at upper bound
                if crsi_val > 65.0:
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