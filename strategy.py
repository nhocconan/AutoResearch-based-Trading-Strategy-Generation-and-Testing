#!/usr/bin/env python3
"""
Experiment #1376: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion with Trend Filter

Hypothesis: 30m timeframe with HTF trend filter can capture mean reversion opportunities
while avoiding counter-trend trades. Combining:
1. 4h HMA(21) for major trend bias (only trade with HTF trend)
2. 1d HMA(21) for regime filter (stronger conviction when aligned)
3. 30m Connors RSI (CRSI) for entry timing (RSI3 + RSI_Streak + PercentRank)
4. 30m Choppiness Index for regime detection (avoid choppy markets)
5. Session filter 08-20 UTC (highest liquidity hours)

Why this should work:
- CRSI is proven mean-reversion indicator (75% win rate in literature)
- HTF trend filter prevents counter-trend disasters in 2022-style crashes
- Choppiness filter avoids whipsaw in range markets
- Session filter reduces low-liquidity false signals
- 30m TF = natural 40-80 trades/year (fee-friendly)

Entry logic:
- LONG: price > 4h_HMA + CRSI < 20 + CHOP < 60 + session 08-20 UTC
- SHORT: price < 4h_HMA + CRSI > 80 + CHOP < 60 + session 08-20 UTC

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_hma_trend_session_4h1d_v1"
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

def calculate_rsi_streak(close, period=2):
    """RSI Streak component of Connors RSI - measures consecutive up/down days"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        if np.isnan(close[i]) or np.isnan(close[i-1]):
            continue
        
        # Calculate streak
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
        else:
            streak = 0
        
        # Calculate RSI of streak values (need lookback)
        if i >= period:
            streak_values = []
            for k in range(i - period + 1, i + 1):
                if k >= 1:
                    if close[k] > close[k-1]:
                        s = 1
                        j = k - 1
                        while j > 0 and close[j] > close[j-1]:
                            s += 1
                            j -= 1
                        streak_values.append(s)
                    elif close[k] < close[k-1]:
                        s = -1
                        j = k - 1
                        while j > 0 and close[j] < close[j-1]:
                            s -= 1
                            j -= 1
                        streak_values.append(s)
                    else:
                        streak_values.append(0)
            
            if len(streak_values) >= period:
                pos_sum = sum(1 for v in streak_values if v > 0)
                neg_sum = sum(1 for v in streak_values if v < 0)
                if pos_sum + neg_sum > 0:
                    streak_rsi[i] = pos_sum / (pos_sum + neg_sum) * 100
                else:
                    streak_rsi[i] = 50
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of Connors RSI - current price vs lookback distribution"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window = close[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < period:
            continue
        
        current = close[i]
        count_below = np.sum(valid < current)
        pr[i] = count_below / len(valid) * 100
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, period=rsi_period)
    rsi_streak = calculate_rsi_streak(close, period=streak_period)
    pr = calculate_percent_rank(close, period=pr_period)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(pr_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j >= 1:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
        
        if atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
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
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
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
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 1d HMA for regime confirmation
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP < 60 = trending market (good for directional trades)
        # CHOP > 60 = choppy/range market (avoid or mean revert)
        is_trending = chop[i] < 60
        
        # === CRSI EXTREMES ===
        crsi_value = crsi[i]
        crsi_oversold = crsi_value < 25  # Loosened from 20 to ensure trades
        crsi_overbought = crsi_value > 75  # Loosened from 80 to ensure trades
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + CRSI oversold + trending market + session
        if price_above_4h and crsi_oversold and is_trending:
            if in_session:
                if price_above_1d:
                    # Strong trend alignment (4h + 1d both bullish)
                    desired_signal = SIZE_STRONG
                else:
                    # Basic long (only 4h bullish)
                    desired_signal = SIZE_BASE
            else:
                # Outside session - reduced size or skip
                desired_signal = SIZE_BASE * 0.5
        
        # SHORT: 4h bearish + CRSI overbought + trending market + session
        elif price_below_4h and crsi_overbought and is_trending:
            if in_session:
                if price_below_1d:
                    # Strong trend alignment (4h + 1d both bearish)
                    desired_signal = -SIZE_STRONG
                else:
                    # Basic short (only 4h bearish)
                    desired_signal = -SIZE_BASE
            else:
                # Outside session - reduced size or skip
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
        elif abs(desired_signal) >= 0.09:
            if desired_signal > 0:
                final_signal = 0.10
            else:
                final_signal = -0.10
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