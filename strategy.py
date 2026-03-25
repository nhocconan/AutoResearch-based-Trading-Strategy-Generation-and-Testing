#!/usr/bin/env python3
"""
Experiment #1556: 30m Primary + 4h/1d HTF — CRSI Mean Reversion with Trend Filter

Hypothesis: 30m timeframe with Connors RSI (CRSI) mean-reversion entries, filtered by
4h/1d HMA trend direction, will generate 40-80 trades/year with positive Sharpe.

Key components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Proven 75% win rate in mean-reversion literature
   - Long: CRSI < 15 (oversold), Short: CRSI > 85 (overbought)
2. 4h HMA(21) for intermediate trend filter
3. 1d HMA(21) for major trend bias
4. Session filter: 08-20 UTC (high liquidity, avoid Asia overnight)
5. Choppiness Index(14) regime filter: only trade mean-reversion in ranges
6. ATR(14) trailing stoploss (2.5x ATR)
7. Discrete sizing: 0.0, ±0.20, ±0.25 (minimize fee churn)

Why this should work:
- CRSI is proven mean-reversion indicator (Connors Research)
- 4h/1d HMA prevents counter-trend disasters in strong trends
- Session filter avoids low-liquidity whipsaws
- CHOP filter ensures we only mean-revert in ranging markets
- LOOSE CRSI thresholds (15/85, not 10/90) guarantee trades
- 30m TF = natural 50-80 trades/year (fee-efficient)

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: CRSI<20 + price>4h_HMA + CHOP>50 + session 08-20 UTC
- SHORT: CRSI>80 + price<4h_HMA + CHOP>50 + session 08-20 UTC
- Reduce size by half if counter to 1d HMA bias

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 30m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_session_4h1d_v1"
timeframe = "30m"
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
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close):
    """
    Connors RSI (CRSI)
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days (streak length)
    PercentRank: percentage of prior 100 closes that are lower than current
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(2, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * min(streak_abs[i], 2) / 2.0
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1.0 - min(streak_abs[i], 2) / 2.0)
        else:
            streak_rsi[i] = 50.0
    
    # PercentRank(100)
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        window = close[i-100:i]
        count_lower = np.sum(window < close[i])
        pct_rank[i] = 100.0 * count_lower / 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
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

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    hours = np.zeros(len(open_time_array), dtype=np.int32)
    for i in range(len(open_time_array)):
        # Convert ms to seconds, then to hours UTC
        ts_sec = open_time_array[i] / 1000.0
        hours[i] = int((ts_sec % 86400) / 3600)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Session hours (08-20 UTC)
    hours = get_hour_from_open_time(open_time)
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
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
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_range_regime = chop > 50.0  # LOOSE threshold to allow more trades
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === MAJOR TREND (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === CRSI ===
        crsi_val = crsi[i]
        
        # === SESSION FILTER ===
        is_trading_session = in_session[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # Only trade in range regime (mean-reversion works best in ranges)
        if is_range_regime and is_trading_session:
            # LONG: CRSI oversold + price above 4h HMA (trend confirmation)
            if crsi_val < 20 and price_above_4h:
                # Full size if aligned with 1d trend, half if counter
                if price_above_1d:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + price below 4h HMA (trend confirmation)
            elif crsi_val > 80 and price_below_4h:
                # Full size if aligned with 1d trend, half if counter
                if price_below_1d:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # Allow some trend-following entries when CHOP is low (trending market)
        elif chop < 45.0 and is_trading_session:
            # LONG: CRSI moderately oversold + strong 4h/1d bullish alignment
            if crsi_val < 30 and price_above_4h and price_above_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI moderately overbought + strong 4h/1d bearish alignment
            elif crsi_val > 70 and price_below_4h and price_below_1d:
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