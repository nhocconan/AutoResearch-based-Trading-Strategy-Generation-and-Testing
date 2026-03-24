#!/usr/bin/env python3
"""
Experiment #1030: 1h Primary + 4h/1d HTF — Simplified Trend Pullback with Session Filter

Hypothesis: After 849 failed experiments with complex regime switching (chop/crsi), 
a SIMPLER approach will work better. Use HTF (4h/1d) for trend direction, 1h for 
pullback entry timing. Key insight from failures: too many filters = 0 trades.

Why this should work:
1. 4h HMA(21) slope gives intermediate trend direction (proven in best strategies)
2. 1d HMA(21) filters major bias (only long if price > 1d HMA, vice versa)
3. 1h RSI(14) pullback: enter on dips in uptrend (RSI 35-55), rallies in downtrend (RSI 45-65)
4. Session filter (08-20 UTC): avoids low liquidity Asian night hours
5. Volume confirmation: volume > 0.8 * 20-bar avg (not too strict)
6. ATR(14) 2.5x trailing stop for risk management

CRITICAL LESSONS FROM FAILURES:
- Experiments #1019, #1021, #1029: Sharpe=0.000 = 0 trades (too many filters)
- Complex chop/crsi regime switching: mostly negative Sharpe
- SIMPLE trend + pullback works better than complex regime detection

Entry conditions (LOOSE to guarantee trades):
- LONG: 4h_HMA_slope>0 + price>1d_HMA*0.98 + RSI(14) 30-60 + session 08-20 UTC
- SHORT: 4h_HMA_slope<0 + price<1d_HMA*1.02 + RSI(14) 40-70 + session 08-20 UTC

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_session_v1"
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback periods"""
    n = len(hma_values)
    slope = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback]
    
    return slope

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
    
    # Calculate 4h HMA slope (trend direction)
    hma_4h_slope = calculate_hma_slope(hma_4h_aligned, lookback=3)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(150, n):
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
        
        if np.isnan(hma_4h_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else True
        
        # === HTF TREND DIRECTION (4h HMA slope) ===
        trend_bull = hma_4h_slope[i] > 0.0005  # Slightly positive slope
        trend_bear = hma_4h_slope[i] < -0.0005  # Slightly negative slope
        
        # === MAJOR BIAS FILTER (1d HMA) ===
        # Loose filter: price within 2% of 1d HMA is ok
        price_vs_1d = close[i] / hma_1d_aligned[i] if hma_1d_aligned[i] > 0 else 1.0
        bias_long = price_vs_1d > 0.98  # Price above or near 1d HMA
        bias_short = price_vs_1d < 1.02  # Price below or near 1d HMA
        
        # === ENTRY LOGIC (PULLBACK IN TREND) ===
        desired_signal = 0.0
        
        # LONG: 4h uptrend + price > 1d HMA + RSI pullback (30-60)
        if trend_bull and bias_long and in_session and vol_ok:
            if 30.0 <= rsi_14[i] <= 60.0:
                # Stronger signal if RSI closer to 35-45 (deeper pullback)
                if 35.0 <= rsi_14[i] <= 50.0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h downtrend + price < 1d HMA + RSI rally (40-70)
        elif trend_bear and bias_short and in_session and vol_ok:
            if 40.0 <= rsi_14[i] <= 70.0:
                # Stronger signal if RSI closer to 50-60 (shallower rally)
                if 50.0 <= rsi_14[i] <= 65.0:
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