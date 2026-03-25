#!/usr/bin/env python3
"""
Experiment #1259: 1h Primary + 4h/12h HTF — CRSI Mean Reversion + HMA Trend + Choppiness Filter

Hypothesis: Current best is 6h KAMA+ROC (Sharpe=0.447). This strategy targets 1h timeframe
with proven pattern: HTF trend direction + CRSI pullback entries + Choppiness regime filter.

Key innovations vs failed strategies:
1. Connors RSI (CRSI) instead of standard RSI - combines RSI(3), RSI-Streak(2), PercentRank(100)
   CRSI catches pullbacks better than RSI(14) in trending markets (75% win rate in literature)
2. Choppiness Index (CHOP) regime filter - only trade when CHOP<50 (trending), avoid ranges
3. Dual HTF confirmation - 12h HMA for major bias, 4h HMA for intermediate trend
4. Session filter 08-20 UTC - avoid low liquidity periods that cause slippage
5. LOOSE CRSI thresholds (25/75 not 10/90) - guarantees 40-80 trades/year

Why this should beat 6h KAMA:
- 1h entries on 4h/12h trend = better entry timing than pure 6h signals
- CRSI mean reversion in trends = buys dips in uptrend, sells rallies in downtrend
- Choppiness filter avoids whipsaw in range markets (major cause of 2022 losses)
- Session filter reduces fee drag from bad fills

Entry logic (LOOSE to guarantee 40-80 trades/year):
- LONG: 12h_HMA rising + price>4h_HMA + CRSI<25 + CHOP<50 + session 08-20 UTC
- SHORT: 12h_HMA falling + price<4h_HMA + CRSI>75 + CHOP<50 + session 08-20 UTC

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_chop_regime_4h12h_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI-Streak(2) + PercentRank(100)) / 3
    
    RSI-Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI for short-term momentum
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI-Streak - measures consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period):i+1]
        if len(streak_window) >= streak_period:
            gains = np.sum(streak_window[streak_window > 0])
            losses = np.abs(np.sum(streak_window[streak_window < 0]))
            if losses == 0:
                streak_rsi[i] = 100.0
            else:
                rs = gains / losses
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # PercentRank - where today's return ranks vs last 100 days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] != 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            count_below = np.sum(valid < returns[i])
            percent_rank[i] = (count_below / len(valid)) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            choppiness[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
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
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC for liquidity) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === TREND DIRECTION (12h HMA slope + 4h HMA bias) ===
        # 12h HMA slope (compare to 3 bars ago for stability)
        hma_12h_slope = 0.0
        if i >= 3 and not np.isnan(hma_12h_aligned[i-3]):
            hma_12h_slope = hma_12h_aligned[i] - hma_12h_aligned[i-3]
        
        # 4h HMA bias
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (Choppiness) ===
        is_trending = choppiness[i] < 50.0  # Below 50 = trending market
        is_ranging = choppiness[i] > 61.8  # Above 61.8 = ranging market
        
        # === CRSI EXTREMES (Mean Reversion in Trend) ===
        crsi_oversold = crsi[i] < 25.0  # Pullback in uptrend
        crsi_overbought = crsi[i] > 75.0  # Rally in downtrend
        
        # === ENTRY LOGIC (LOOSE - guarantee 40-80 trades/year) ===
        desired_signal = 0.0
        
        # LONG: 12h HMA rising + price>4h_HMA + CRSI oversold + trending + session
        if hma_12h_slope > 0 and price_above_4h and crsi_oversold and is_trending:
            if in_session:
                if crsi[i] < 15.0:
                    desired_signal = SIZE_STRONG  # Very oversold
                else:
                    desired_signal = SIZE_BASE  # Moderately oversold
        
        # SHORT: 12h HMA falling + price<4h_HMA + CRSI overbought + trending + session
        elif hma_12h_slope < 0 and price_below_4h and crsi_overbought and is_trending:
            if in_session:
                if crsi[i] > 85.0:
                    desired_signal = -SIZE_STRONG  # Very overbought
                else:
                    desired_signal = -SIZE_BASE  # Moderately overbought
        
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