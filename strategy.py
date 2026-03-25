#!/usr/bin/env python3
"""
Experiment #1261: 15m Primary + 1h/4h/1d HTF — Regime-Filtered Mean Reversion

Hypothesis: 15m strategies have failed due to either 0 trades or too many trades (fee drag).
This strategy uses HTF trend direction (4h/1d HMA) with 15m RSI mean-reversion entries,
filtered by Choppiness Index regime and UTC session timing. Key innovations:

1. 4h HMA(21) for primary trend direction (aligned properly via mtf_data)
2. 1d HMA(21) for major regime bias (only trade with daily trend)
3. 15m RSI(7) for oversold/overbought entries within trend (Connors-style)
4. Choppiness Index(14) < 50 = trending regime (avoid range chop)
5. Session filter: UTC hours 00-12 (London/NY overlap, best crypto liquidity)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.15 base, 0.20 strong (smaller for 15m frequency)

Why this should work on 15m:
- HTF trend filter = only 40-100 trades/year (fee-friendly)
- RSI(7) extremes = catches intraday pullbacks in trend
- CHOP filter = avoids whipsaw in ranging markets
- Session filter = trades during highest volume periods
- Small position size (0.15-0.20) = limits drawdown on 77% BTC crash

Entry logic (selective but not impossible):
- LONG: 4h_HMA rising + 1d_price > 1d_HMA + RSI(7) < 30 + CHOP < 50 + UTC 00-12
- SHORT: 4h_HMA falling + 1d_price < 1d_HMA + RSI(7) > 70 + CHOP < 50 + UTC 00-12

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher TF frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_rsi_meanrev_4h1d_session_v1"
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

def calculate_rsi(close, period=7):
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
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest == lowest:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def get_utc_hour(prices, i):
    """Extract UTC hour from open_time column"""
    try:
        ts = prices['open_time'].iloc[i]
        if isinstance(ts, (int, np.integer)):
            ts = ts / 1000.0
        if hasattr(ts, 'timestamp'):
            ts = ts.timestamp()
        from datetime import datetime
        dt = datetime.utcfromtimestamp(ts)
        return dt.hour
    except:
        return 12  # Default to trading hours if parsing fails

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Also calculate 15m HMA for local trend confirmation
    hma_15m = calculate_hma(close, period=21)
    
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
    entry_bar = 0
    
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
        
        if np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC hours 00-12 preferred) ===
        utc_hour = get_utc_hour(prices, i)
        in_session = (utc_hour >= 0 and utc_hour <= 12)
        
        # === TREND DIRECTION (4h HMA slope + 1d HMA bias) ===
        # 4h HMA slope (compare to 3 bars ago for stability)
        hma_4h_slope = 0.0
        if i >= 3 and not np.isnan(hma_4h_aligned[i-3]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-3]
        
        # 1d HMA bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 15m price vs 15m HMA for local confirmation
        price_above_15m = close[i] > hma_15m[i]
        price_below_15m = close[i] < hma_15m[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # Below 50 = trending market
        is_chopping = chop > 61.8  # Above 61.8 = choppy market
        
        # === MOMENTUM (RSI) ===
        rsi = rsi_7[i]
        
        # === ENTRY LOGIC (Selective but achievable) ===
        desired_signal = 0.0
        
        # LONG: 4h HMA rising + 1d bullish + RSI oversold + trending regime + session
        if hma_4h_slope > 0 and price_above_1d:
            if rsi < 30.0 and is_trending:
                if in_session:
                    if rsi < 20.0:
                        desired_signal = SIZE_STRONG  # Deep oversold
                    else:
                        desired_signal = SIZE_BASE  # Standard oversold
        
        # SHORT: 4h HMA falling + 1d bearish + RSI overbought + trending regime + session
        elif hma_4h_slope < 0 and price_below_1d:
            if rsi > 70.0 and is_trending:
                if in_session:
                    if rsi > 80.0:
                        desired_signal = -SIZE_STRONG  # Deep overbought
                    else:
                        desired_signal = -SIZE_BASE  # Standard overbought
        
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
        
        # Time-based exit after 48 bars (12 hours on 15m)
        time_exit = False
        if in_position and (i - entry_bar) > 48:
            time_exit = True
        
        if stoploss_triggered or time_exit:
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
                entry_bar = i
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
                entry_bar = 0
        
        signals[i] = final_signal
    
    return signals