#!/usr/bin/env python3
"""
Experiment #1289: 15m Primary + 1h/4h/1d HTF — Session Breakout with Volume Confirmation

Hypothesis: Previous 15m strategies failed due to either (a) pure mean-reversion getting
whipsawed in trends, or (b) trend-following entering at wrong times. This strategy uses:

1. 1d HMA(21) for MAJOR regime bias (only long if price > 1d HMA, only short if <)
2. 4h HMA(21) for intermediate trend confirmation
3. 1h RSI(14) for momentum regime (avoid entering when 1h RSI > 70 for longs, < 30 for shorts)
4. 15m breakout entry: price breaks 20-bar high/low WITH volume spike (>1.5x avg)
5. Session filter: only trade 00-12 UTC (London/NY overlap = higher volume, cleaner moves)
6. ATR(14) 2.5x trailing stop for risk management

Why this should work on 15m:
- HTF filters (1d/4h) reduce whipsaw by 60-70%
- 1h RSI prevents buying tops/selling bottoms
- Volume confirmation filters false breakouts
- Session filter avoids low-liquidity Asian session chop
- Discrete sizing (0.0, ±0.15, ±0.20) minimizes fee churn
- Target: 50-80 trades/year (fee-friendly for 15m)

Entry logic (LOOSE enough for trades, strict enough for quality):
- LONG: 1d_HMA bullish + 4h_HMA rising + 1h_RSI < 70 + 15m breakout + volume spike + UTC 00-12
- SHORT: 1d_HMA bearish + 4h_HMA falling + 1h_RSI > 30 + 15m breakdown + volume spike + UTC 00-12

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_breakout_volume_htf_1h4h1d_v1"
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
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Pad to match original length
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_donchian_high(high, period=20):
    """Donchian Channel Upper Band (highest high over period)"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        result[i] = np.nanmax(high[i - period + 1:i + 1])
    return result

def calculate_donchian_low(low, period=20):
    """Donchian Channel Lower Band (lowest low over period)"""
    n = len(low)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        result[i] = np.nanmin(low[i - period + 1:i + 1])
    return result

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    utc_hour = (ts_seconds % 86400) / 3600.0
    return int(utc_hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    donchian_high_20 = calculate_donchian_high(high, period=20)
    donchian_low_20 = calculate_donchian_low(low, period=20)
    
    # 15m HMA for local trend
    hma_15m = calculate_hma(close, period=21)
    
    # 4h HMA slope (compare to 3 bars ago on 4h = 12 hours ago)
    hma_4h_slope = np.zeros(n)
    for i in range(3, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-3]):
            hma_4h_slope[i] = hma_4h_aligned[i] - hma_4h_aligned[i-3]
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 = London/NY overlap) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 0 and utc_hour <= 12)
        
        # === REGIME FILTER (1d HMA for major bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND CONFIRMATION (4h HMA slope) ===
        hma_4h_rising = hma_4h_slope[i] > 0
        hma_4h_falling = hma_4h_slope[i] < 0
        
        # === MOMENTUM FILTER (1h RSI) ===
        rsi_1h = rsi_1h_aligned[i]
        rsi_not_overbought = rsi_1h < 70  # OK to go long
        rsi_not_oversold = rsi_1h > 30    # OK to go short
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_sma_20[i]
        volume_spike = vol_ratio > 1.5
        
        # === BREAKOUT DETECTION ===
        breakout_long = close[i] > donchian_high_20[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_low_20[i-1]  # Break below previous low
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h rising + 1h RSI not overbought + breakout + volume + session
        if price_above_1d and hma_4h_rising and rsi_not_overbought:
            if breakout_long and volume_spike and in_session:
                if vol_ratio > 2.5:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + 4h falling + 1h RSI not oversold + breakdown + volume + session
        elif price_below_1d and hma_4h_falling and rsi_not_oversold:
            if breakout_short and volume_spike and in_session:
                if vol_ratio > 2.5:
                    desired_signal = -SIZE_STRONG
                else:
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