#!/usr/bin/env python3
"""
Experiment #1189: 15m Primary + 1h/1d HTF — Camarilla Pivot Mean Reversion

Hypothesis: After 4 failed 15m experiments (all 0 trades from over-filtering), 
this strategy uses LOOSE entry conditions based on Camarilla pivot levels which
naturally create mean reversion opportunities.

Key changes from failed 15m experiments:
1. NO session filter (causes 0 trades in #1177, #1185)
2. Faster RSI(7) instead of RSI(14) for more signals
3. Camarilla levels = natural mean reversion targets (price extends → reverts)
4. 1h HMA for trend bias (not 4h, less lag for 15m entries)
5. Discrete sizing 0.15-0.20 (smaller for 15m frequency)

Entry logic (LOOSE to guarantee trades):
- LONG: 1h HMA bullish + 15m price touches S2/S3 + RSI(7) < 40
- SHORT: 1h HMA bearish + 15m price touches R2/R3 + RSI(7) > 60
- Breakout: Price breaks R4/S4 with momentum = trend follow

Target: 40-100 trades/year, Sharpe>0.5, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_pivot_rsi_meanrev_1h1d_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_camarilla_pivots(high, low, close, prev_close):
    """
    Camarilla Pivot Points - mean reversion levels
    R3/R4 = resistance, S3/S4 = support
    Price tends to revert from R3/S3, breakout at R4/S4
    """
    n = len(close)
    pivot_range = prev_close[1:] - np.roll(prev_close, 1)[1:]
    pivot_range[0] = high[1] - low[1]
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        pp[i] = (h + l + c) / 3.0
        
        range_val = h - l
        r3[i] = c + range_val * 1.1 / 12.0
        r4[i] = c + range_val * 1.1 / 2.0
        s3[i] = c - range_val * 1.1 / 12.0
        s4[i] = c - range_val * 1.1 / 2.0
    
    return pp, r3, r4, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Get daily OHLC for Camarilla pivots
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate daily Camarilla pivots
    pp_1d, r3_1d, r4_1d, s3_1d, s4_1d = calculate_camarilla_pivots(
        daily_high, daily_low, daily_close, np.roll(daily_close, 1)
    )
    
    # Align daily pivots to 15m
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    
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
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(pp_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1h HMA) ===
        price_above_1h = close[i] > hma_1h_aligned[i]
        price_below_1h = close[i] < hma_1h_aligned[i]
        
        # === CAMARILLA LEVELS ===
        pp = pp_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        
        # Check if price is near pivot levels (within 0.3% tolerance)
        tolerance = 0.003  # 0.3%
        
        near_s3 = abs(close[i] - s3) / s3 < tolerance if not np.isnan(s3) else False
        near_s4 = close[i] < s4 if not np.isnan(s4) else False
        near_r3 = abs(close[i] - r3) / r3 < tolerance if not np.isnan(r3) else False
        near_r4 = close[i] > r4 if not np.isnan(r4) else False
        
        below_s3 = close[i] < s3 if not np.isnan(s3) else False
        above_r3 = close[i] > r3 if not np.isnan(r3) else False
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        rsi_fast = rsi_7[i]
        rsi_slow = rsi_14[i]
        
        # LONG: Mean reversion at support
        # Condition 1: Price at/below S3 + RSI oversold
        if below_s3 and rsi_fast < 40:
            if price_above_1h:
                desired_signal = SIZE_STRONG  # Trend + MR confluence
            else:
                desired_signal = SIZE_BASE  # Pure mean reversion
        
        # Condition 2: Breakout above R4 with momentum (trend follow)
        elif near_r4 and rsi_fast > 55 and rsi_slow > 50:
            if price_above_1h:
                desired_signal = SIZE_STRONG  # Breakout with trend
        
        # SHORT: Mean reversion at resistance
        # Condition 1: Price at/above R3 + RSI overbought
        if desired_signal == 0.0:  # Only if no long signal
            if above_r3 and rsi_fast > 60:
                if price_below_1h:
                    desired_signal = -SIZE_STRONG  # Trend + MR confluence
                else:
                    desired_signal = -SIZE_BASE  # Pure mean reversion
            
            # Condition 2: Breakout below S4 with momentum (trend follow)
            elif near_s4 and rsi_fast < 45 and rsi_slow < 50:
                if price_below_1h:
                    desired_signal = -SIZE_STRONG  # Breakout with trend
        
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