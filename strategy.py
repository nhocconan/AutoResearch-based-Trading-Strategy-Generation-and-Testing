#!/usr/bin/env python3
"""
Experiment #1625: 15m Primary + 4h/1d HTF — Session-Aware Mean Reversion with Trend Filter

Hypothesis: 15m timeframe with strict 4h trend bias + session filter + Connors RSI mean reversion
can capture intraday swings while avoiding fee drag. Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — faster than standard RSI
2. 4h HMA slope for trend bias (only trade long if 4h HMA rising, short if falling)
3. Session filter: 00-12 UTC only (London/NY overlap = highest liquidity)
4. Bollinger Band width filter: only trade when BB width > 20th percentile (avoid dead chop)
5. Discrete sizes: 0.15 base, 0.20 strong (smaller for 15m frequency)
6. 2.0x ATR stoploss (tighter for lower TF)

Why this might beat failed 15m strategies (#1617, #1621):
- Stricter confluence (4 filters vs 2-3 in failed attempts)
- Session filter eliminates low-liquidity Asian session whipsaws
- CRSI more responsive than RSI(14) for 15m entries
- BB width filter avoids dead chop periods

Target: Sharpe>0.6, trades≥40/train, trades≥5/test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_session_4h1d_bbwidth_v1"
timeframe = "15m"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    More responsive than standard RSI for mean reversion
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
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
        
        # Convert streak to RSI-like value (0-100)
        if streak > 0:
            streak_rsi[i] = min(100, 50 + streak * 25)
        elif streak < 0:
            streak_rsi[i] = max(0, 50 + streak * 25)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        if not np.isnan(close[i]):
            window = close[i - rank_period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                count_below = np.sum(valid < close[i])
                percent_rank[i] = 100.0 * count_below / len(valid)
    
    # Combine
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_bb_width(upper, lower, mid, period=20):
    """Bollinger Band Width as % of mid"""
    n = len(mid)
    bb_width = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(upper[i]) and not np.isnan(lower[i]) and mid[i] > 1e-10:
            bb_width[i] = (upper[i] - lower[i]) / mid[i] * 100.0
    return bb_width

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)"""
    n = len(hma_values)
    slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i - lookback]):
            slope[i] = hma_values[i] - hma_values[i - lookback]
    return slope

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds)"""
    # open_time is in milliseconds since epoch
    hour = (open_time // 1000 // 3600) % 24
    return hour

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
    
    hma_4h_slope_raw = calculate_hma_slope(hma_4h_raw, lookback=3)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_3 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_mid, period=20)
    
    # Calculate BB width percentile (20-period rolling)
    bb_width_pct = np.full(n, np.nan, dtype=np.float64)
    for i in range(20, n):
        window = bb_width[i-20:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0 and not np.isnan(bb_width[i]):
            bb_width_pct[i] = np.sum(valid < bb_width[i]) / len(valid) * 100.0
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
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
        
        if np.isnan(crsi_3[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 0 <= hour <= 12
        
        # === TREND BIAS (4h HMA slope) ===
        hma_4h_slope = hma_4h_slope_aligned[i]
        trend_bullish = hma_4h_slope > 0
        trend_bearish = hma_4h_slope < 0
        
        # === 1D REGIME FILTER ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY FILTER (BB Width > 20th percentile) ===
        vol_expanding = bb_width_pct[i] > 20.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_val = crsi_3[i]
        crsi_oversold = crsi_val < 15  # Very oversold
        crsi_overbought = crsi_val > 85  # Very overbought
        crsi_neutral_low = crsi_val < 30
        crsi_neutral_high = crsi_val > 70
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.005
        bb_touch_upper = close[i] >= bb_upper[i] * 0.995
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: 4h uptrend + CRSI oversold + BB touch + session + vol
        if trend_bullish and price_above_1d and in_session and vol_expanding:
            if crsi_oversold and bb_touch_lower:
                desired_signal = SIZE_STRONG
            elif crsi_neutral_low and bb_touch_lower:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h downtrend + CRSI overbought + BB touch + session + vol
        elif trend_bearish and price_below_1d and in_session and vol_expanding:
            if crsi_overbought and bb_touch_upper:
                desired_signal = -SIZE_STRONG
            elif crsi_neutral_high and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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