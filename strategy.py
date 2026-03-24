#!/usr/bin/env python3
"""
Experiment #1076: 30m Primary + 4h/1d HTF — HMA Trend Bias + RSI Pullback + Volume + Session

Hypothesis: Simple is better. Complex regime-switching (Choppiness, Connors RSI) failed in #1068, #1070.
Instead use proven pattern: HTF trend direction + LTF pullback entry + volume confirmation + session filter.

Key innovations:
1. 4h HMA(21) slope for trend bias (long only when 4h HMA rising)
2. 1d HMA(21) for major trend filter (only long when price > 1d HMA)
3. 30m RSI(14) pullback entries (RSI < 35 for long, RSI > 65 for short)
4. Volume spike confirmation (30m volume > 1.5x 20-bar average)
5. Session filter: 08:00-20:00 UTC only (high liquidity, avoid Asian session whipsaws)
6. ATR(14) 2.5x trailing stoploss

Why this should work:
- 4h trend filter avoids counter-trend trades (main failure of simple RSI strategies)
- 1d HMA ensures we're aligned with major trend
- RSI pullback captures dips in uptrends (high win rate pattern)
- Volume filter ensures real moves, not noise
- Session filter reduces whipsaws during low-liquidity hours
- 30m timeframe = ~50-80 trades/year target (fee-friendly)

Entry conditions (LOOSE enough for trades):
- LONG: 4h_HMA_slope > 0 + price > 1d_HMA + RSI(14) < 40 + volume > 1.3x avg + session 08-20 UTC
- SHORT: 4h_HMA_slope < 0 + price < 1d_HMA + RSI(14) > 60 + volume > 1.3x avg + session 08-20 UTC

Target: Sharpe > 0.45, trades >= 40/year, DD > -40%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_trend_rsi_pullback_volume_session_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_volume_spike(volume, period=20, threshold=1.3):
    """Detect volume spikes vs recent average"""
    n = len(volume)
    if n < period:
        return np.zeros(n, dtype=bool)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_avg)
    spike[:period] = False
    return spike

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    
    # Calculate 4h HMA slope (direction of trend)
    hma_4h_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-5]):
            hma_4h_slope[i] = hma_4h_aligned[i] - hma_4h_aligned[i-5]
        else:
            hma_4h_slope[i] = 0.0
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    volume_spike = calculate_volume_spike(volume, period=20, threshold=1.3)
    
    # Session filter: 08:00 - 20:00 UTC only
    session_valid = np.zeros(n, dtype=bool)
    for i in range(n):
        hour = get_session_hour(open_time[i])
        session_valid[i] = (hour >= 8) and (hour < 20)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
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
        
        # === TREND BIAS (4h HMA slope + 1d HMA position) ===
        trend_bull = (hma_4h_slope[i] > 0) and (close[i] > hma_1d_aligned[i])
        trend_bear = (hma_4h_slope[i] < 0) and (close[i] < hma_1d_aligned[i])
        
        # === SESSION FILTER ===
        in_session = session_valid[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume_spike[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: uptrend + RSI pullback + volume + session
        if trend_bull and in_session:
            if rsi_14[i] < 35.0 and vol_confirmed:
                desired_signal = SIZE_STRONG
            elif rsi_14[i] < 40.0:
                desired_signal = SIZE_BASE
        
        # SHORT: downtrend + RSI rally + volume + session
        elif trend_bear and in_session:
            if rsi_14[i] > 65.0 and vol_confirmed:
                desired_signal = -SIZE_STRONG
            elif rsi_14[i] > 60.0:
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