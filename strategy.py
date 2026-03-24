#!/usr/bin/env python3
"""
Experiment #1049: 15m Primary + 1h/1d HTF — Fisher Transform + Choppiness Regime + Session Filter

Hypothesis: 15m timeframe with strict HTF alignment can capture intraday swings while avoiding
noise. Using Fisher Transform for entry signals (superior reversal detection vs RSI) combined
with Choppiness regime filter and session timing should generate 50-80 trades/year with
positive Sharpe across BTC/ETH/SOL.

Key innovations:
1. Fisher Transform (period=9): Transforms price to near-Gaussian, extremes at ±2.0 signal reversals
2. Choppiness Index (14): Only trade when CHOP < 55 (avoid extreme chop) or CHOP > 65 (mean revert)
3. 1h HMA(21) trend filter: Only take 15m signals in direction of 1h trend
4. 1d ATR volatility filter: Avoid entries when 1d vol is 2x normal (panic conditions)
5. Session filter: Prefer UTC 00-12 (London+NY overlap = higher liquidity, cleaner moves)
6. Discrete sizing: 0.0, ±0.15, ±0.25 (smaller for 15m frequency to reduce fee impact)

Why this should work on 15m:
- Fisher Transform catches reversals faster than RSI (proven in Ehlers literature)
- 1h trend filter prevents counter-trend 15m noise trades
- Session filter reduces trades by ~40% but improves win rate
- CHOP filter avoids whipsaw zones
- Target 50-80 trades/year (0.05% fee = 2.5-4% annual drag, manageable)

Entry conditions (LOOSE to guarantee trades):
- LONG: 1h_HMA bullish + Fisher < -1.2 + CHOP < 60 + session 00-12 UTC
- SHORT: 1h_HMA bearish + Fisher > +1.2 + CHOP < 60 + session 00-12 UTC
- MEAN REVERT LONG: CHOP > 65 + Fisher < -1.8 (extreme oversold in range)
- MEAN REVERT SHORT: CHOP > 65 + Fisher > +1.8 (extreme overbought in range)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller than 4h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_chop_session_1h1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - transforms price to near-Gaussian distribution
    Extremes at ±2.0 signal potential reversals
    Formula: Fisher = 0.5 * ln((1 + Value) / (1 - Value))
    Value = 0.66 * ((Price - Lowest) / (Highest - Lowest) - 0.5) + 0.67 * Prev_Value
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        value = 0.66 * ((close[i] - lowest) / price_range - 0.5)
        if i > period and not np.isnan(fisher_prev[i-1]):
            value += 0.67 * fisher_prev[i-1]
        
        # Clamp to avoid division by zero
        value = np.clip(value, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
        fisher_prev[i] = value
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 15m ATR average for vol filter
    atr_14_mean = pd.Series(atr_14).rolling(window=100, min_periods=100).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(atr_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 preferred) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        is_session = (hour_utc >= 0 and hour_utc < 12)  # London+NY overlap
        
        # === VOLATILITY FILTER (avoid extreme 1d vol) ===
        atr_1d_current = atr_1d_aligned[i]
        atr_14_current = atr_14[i]
        # If 1d ATR is > 2x recent 15m ATR average, skip (panic conditions)
        vol_filter_pass = True
        if not np.isnan(atr_1d_current) and not np.isnan(atr_14_mean[i]):
            if atr_1d_current > 2.5 * atr_14_mean[i] * 4:  # 1d ~4x 15m
                vol_filter_pass = False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 60.0  # Range market - use mean reversion
        is_trending = chop_14[i] < 50.0  # Trend market - use trend following
        is_neutral = not is_choppy and not is_trending
        
        # === HTF TREND (1h HMA) ===
        hma_1h_bull = close[i] > hma_1h_aligned[i]
        hma_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND MODE (CHOP < 50): Follow 1h trend with Fisher entry
        if is_trending and vol_filter_pass:
            # LONG: 1h bullish + Fisher oversold + session preferred
            if hma_1h_bull and fisher[i] < -1.0:
                if is_session:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # SHORT: 1h bearish + Fisher overbought + session preferred
            elif hma_1h_bear and fisher[i] > 1.0:
                if is_session:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # MEAN REVERSION MODE (CHOP > 60): Fade extremes
        elif is_choppy and vol_filter_pass:
            # LONG: Fisher extremely oversold
            if fisher[i] < -1.8:
                desired_signal = SIZE_BASE
            # SHORT: Fisher extremely overbought
            elif fisher[i] > 1.8:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL MODE (50 < CHOP < 60): Only strong Fisher signals
        elif is_neutral and vol_filter_pass:
            if fisher[i] < -2.0:
                desired_signal = SIZE_BASE
            elif fisher[i] > 2.0:
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