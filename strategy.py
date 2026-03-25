#!/usr/bin/env python3
"""
Experiment #1256: 30m Primary + 4h/1d HTF — Trend Pullback with Choppiness Regime Filter

Hypothesis: The 6h KAMA+ROC strategy (Sharpe=0.447) works because it combines trend 
with momentum confirmation. For 30m, I'll use a DIFFERENT approach: trend-following 
WITH mean-reversion entries (buy pullbacks in uptrend, sell rallies in downtrend).

Key innovations vs failed strategies:
1. 4h HMA(21) for primary trend direction (not 12h - too slow for 30m entries)
2. 1d HMA(21) for regime bias (only trade WITH daily trend)
3. 30m RSI(7) for pullback entries (RSI 30-50 in uptrend = buy dip, not RSI extremes)
4. Choppiness(14) < 45 to confirm TRENDING regime (pullbacks work best in trends)
5. Session filter: 08-20 UTC (avoid Asian session whipsaw, higher liquidity)
6. ATR(14) 2.5x trailing stoploss

Why this should beat 6h KAMA:
- 30m captures more entry opportunities within 4h trend moves
- Pullback entries have better risk/reward than breakout entries
- Choppiness filter avoids mean-reversion traps in ranging markets
- Session filter reduces false signals during low-liquidity hours

Entry logic (3+ confluence required):
- LONG: 4h_HMA rising + price>4h_HMA + 1d_HMA bullish + RSI(7) 30-50 + CHOP<45 + session 08-20
- SHORT: 4h_HMA falling + price<4h_HMA + 1d_HMA bearish + RSI(7) 50-70 + CHOP<45 + session 08-20

Target: Sharpe>0.5, trades>=40/year train, trades>=5 test, DD>-35%
Timeframe: 30m
Size: 0.20 base, 0.30 strong (discrete levels to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_trend_pullback_rsi_chop_4h1d_v1"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market ranging vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar (True Range, not smoothed)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

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
    rsi_7 = calculate_rsi(close, period=7)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Also calculate 30m HMA for local trend confirmation
    hma_30m = calculate_hma(close, period=21)
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(chop_14[i]):
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
        
        if np.isnan(hma_30m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === TREND DIRECTION (4h HMA slope + 1d HMA bias) ===
        # 4h HMA slope (compare to 2 bars ago for stability)
        hma_4h_slope = 0.0
        if i >= 2 and not np.isnan(hma_4h_aligned[i-2]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-2]
        
        # 1d HMA bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 4h price vs 4h HMA
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 30m price vs 30m HMA for local confirmation
        price_above_30m = close[i] > hma_30m[i]
        price_below_30m = close[i] < hma_30m[i]
        
        # === CHOPPINESS REGIME FILTER ===
        chop = chop_14[i]
        is_trending = chop < 45.0  # Trending market (pullbacks work best)
        
        # === RSI PULLBACK ENTRY ===
        rsi = rsi_7[i]
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: 4h HMA rising + price>4h_HMA + 1d bullish + RSI pullback + trending + session
        if hma_4h_slope > 0 and price_above_4h and price_above_1d and is_trending and in_session:
            # RSI pullback in uptrend (30-50 = dip buying opportunity)
            if 30.0 <= rsi <= 50.0:
                if price_above_30m:  # Extra confirmation: price above 30m HMA
                    if rsi <= 40.0:
                        desired_signal = SIZE_STRONG  # Deeper pullback
                    else:
                        desired_signal = SIZE_BASE  # Standard pullback
        
        # SHORT: 4h HMA falling + price<4h_HMA + 1d bearish + RSI rally + trending + session
        elif hma_4h_slope < 0 and price_below_4h and price_below_1d and is_trending and in_session:
            # RSI rally in downtrend (50-70 = selling opportunity)
            if 50.0 <= rsi <= 70.0:
                if price_below_30m:  # Extra confirmation: price below 30m HMA
                    if rsi >= 60.0:
                        desired_signal = -SIZE_STRONG  # Stronger rally
                    else:
                        desired_signal = -SIZE_BASE  # Standard rally
        
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