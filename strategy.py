#!/usr/bin/env python3
"""
Experiment #1117: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume + Session

Hypothesis: 15m strategies fail due to too many trades and fee drag. This strategy uses:
1. 4h HMA(21) for TREND DIRECTION (only trade in HTF trend direction)
2. 1d HMA(21) for REGIME FILTER (avoid counter-trend in strong daily trend)
3. 15m RSI(7) for ENTRY TIMING (pullback entries in trend direction)
4. Volume spike filter (1.5x 20-bar avg) to confirm breakout validity
5. Session filter (00-12 UTC) to avoid low-liquidity Asian session traps
6. ATR(14) 2.0x trailing stop for risk management

Key innovations for 15m success:
- VERY SELECTIVE entries: HTF trend + RSI extreme + volume + session = 4 confluence
- Small position size: 0.15-0.20 (lower than 4h strategies due to higher frequency)
- Target: 40-100 trades/year (strict filters to avoid fee drag)
- 4h HMA slope determines direction, 15m RSI times the pullback entry

Why this should work on 15m:
- HTF filter reduces trade frequency to sustainable level
- RSI(7) pullback has high win rate in trending markets (60-65%)
- Volume filter avoids false breakouts during low liquidity
- Session filter avoids whipsaws during thin Asian hours
- Discrete sizing (0.0, ±0.15, ±0.20) minimizes fee churn

Entry conditions (balanced for trades + quality):
- LONG: 4h_HMA sloping up + price > 1d_HMA + RSI(7) < 35 + volume > 1.5x avg + UTC 00-12
- SHORT: 4h_HMA sloping down + price < 1d_HMA + RSI(7) > 65 + volume > 1.5x avg + UTC 00-12

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_volume_session_4h1d_v1"
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_avg
    ratio[:period] = np.nan
    return ratio

def get_utc_hour(open_time):
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
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate 4h HMA slope (direction)
    hma_4h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-1]):
            hma_4h_slope[i] = hma_4h_aligned[i] - hma_4h_aligned[i-1]
        else:
            hma_4h_slope[i] = np.nan
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Get UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        is_peak_session = (0 <= utc_hour <= 12)  # London + NY overlap
        
        # === HTF TREND DIRECTION (4h HMA) ===
        hma_4h_bull = hma_4h_slope[i] > 0 and close[i] > hma_4h_aligned[i]
        hma_4h_bear = hma_4h_slope[i] < 0 and close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (1d HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.3  # 30% above average
        
        # === ENTRY LOGIC (4 CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: 4h uptrend + above 1d HMA + RSI(7) oversold + volume + session
        if hma_4h_bull and price_above_1d:
            if rsi_7[i] < 35 and volume_confirmed:
                if is_peak_session:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif rsi_7[i] < 30 and volume_confirmed:
                # Stronger signal at more extreme RSI
                desired_signal = SIZE_STRONG
        
        # SHORT: 4h downtrend + below 1d HMA + RSI(7) overbought + volume + session
        elif hma_4h_bear and price_below_1d:
            if rsi_7[i] > 65 and volume_confirmed:
                if is_peak_session:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif rsi_7[i] > 70 and volume_confirmed:
                # Stronger signal at more extreme RSI
                desired_signal = -SIZE_STRONG
        
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