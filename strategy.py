#!/usr/bin/env python3
"""
Experiment #1525: 15m Primary + 4h/1d HTF — Session-Filtered Mean Reversion

Hypothesis: 15m timeframe can work with VERY selective entries using:
1. 1d HMA(21) for major trend bias (avoid counter-trend)
2. 4h ATR ratio for volatility regime (high vol = mean revert, low vol = trend)
3. 15m RSI(7) for fast entry timing (more sensitive than RSI14)
4. Session filter: only trade 00-12 UTC (London+NY overlap = best liquidity)
5. Volume confirmation: taker_buy_volume ratio for conviction

Why this should work on 15m:
- Session filter cuts trades by ~50% (only 12h/day vs 24h)
- RSI(7) extremes happen frequently enough on 15m (guarantees trades)
- 1d/4h HTF filters prevent whipsaw in strong trends
- Discrete sizing 0.15-0.25 minimizes fee churn at higher frequency

Entry logic (LOOSE enough to guarantee trades):
- LONG: 1d_HMA bullish + RSI7<30 + session 00-12 UTC + volume confirmation
- SHORT: 1d_HMA bearish + RSI7>70 + session 00-12 UTC + volume confirmation
- Add mean-reversion mode when 4h ATR ratio > 1.5 (vol spike = revert)

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%, trades/year <100
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi7_hma_4h1d_v1"
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

def calculate_session_mask(open_time):
    """
    Session filter: 00-12 UTC only (London + NY overlap for crypto)
    Returns boolean array where True = trade allowed
    """
    # open_time is in milliseconds since epoch
    hours = pd.to_datetime(open_time, unit='ms').dt.hour.values
    return (hours >= 0) & (hours < 12)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    atr_4h_raw = calculate_atr(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
    # Calculate 15m indicators
    hma_21 = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume ratio (taker buy / total volume)
    volume_ratio = np.full(n, np.nan, dtype=np.float64)
    mask = volume > 0
    volume_ratio[mask] = taker_buy_volume[mask] / volume[mask]
    
    # Session mask
    session_allowed = calculate_session_mask(open_time)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d TREND BIAS ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h VOLATILITY REGIME (ATR ratio) ===
        # Compare current 4h ATR to recent average
        atr_4h = atr_4h_aligned[i]
        atr_4h_avg = np.nanmean(atr_4h_aligned[max(0, i-48):i])  # Last 12h of 4h data
        vol_ratio = atr_4h / atr_4h_avg if atr_4h_avg > 0 and not np.isnan(atr_4h_avg) else 1.0
        high_vol_regime = vol_ratio > 1.3  # Vol spike = mean reversion mode
        low_vol_regime = vol_ratio < 0.8   # Low vol = trend mode
        
        # === 15m RSI ===
        rsi_fast = rsi_7[i]
        rsi_slow = rsi_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_conf = volume_ratio[i]
        vol_bullish = vol_conf > 0.55  # More buying pressure
        vol_bearish = vol_conf < 0.45  # More selling pressure
        
        # === SESSION FILTER ===
        in_session = session_allowed[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG entries
        if price_above_1d and in_session:
            # Mean reversion: RSI7 very oversold
            if rsi_fast < 28:
                if high_vol_regime or vol_bullish:
                    desired_signal = SIZE_BASE
            
            # Trend pullback: RSI7 moderately oversold + volume confirmation
            elif rsi_fast < 40 and vol_bullish:
                if low_vol_regime:
                    desired_signal = SIZE_STRONG
        
        # SHORT entries
        elif price_below_1d and in_session:
            # Mean reversion: RSI7 very overbought
            if rsi_fast > 72:
                if high_vol_regime or vol_bearish:
                    desired_signal = -SIZE_BASE
            
            # Trend pullback: RSI7 moderately overbought + volume confirmation
            elif rsi_fast > 60 and vol_bearish:
                if low_vol_regime:
                    desired_signal = -SIZE_STRONG
        
        # Counter-trend only in high vol regime (mean reversion favored)
        if high_vol_regime and in_session:
            if rsi_fast < 20:  # Extreme oversold
                desired_signal = max(desired_signal, SIZE_BASE * 0.5)
            elif rsi_fast > 80:  # Extreme overbought
                desired_signal = min(desired_signal, -SIZE_BASE * 0.5)
        
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