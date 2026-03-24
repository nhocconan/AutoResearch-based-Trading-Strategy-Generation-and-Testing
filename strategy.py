#!/usr/bin/env python3
"""
Experiment #965: 15m Primary + 4h/1d HTF — Session-Filtered Trend Pullback

Hypothesis: 15m timeframe with strict session filter (00-12 UTC) + 4h HMA trend
+ 1d regime bias + RSI(7) pullback entries will generate selective, high-quality
trades while avoiding the fee drag that kills lower TF strategies.

Key innovations:
1. SESSION FILTER: Only trade 00-12 UTC (London+NY overlap = 75% of crypto volume)
2. 4h HMA(21) for intermediate trend direction (aligned properly via mtf_data)
3. 1d regime: close > open = bull bias, close < open = bear bias
4. 15m RSI(7) pullback entries: RSI 30-45 for long, 55-70 for short (NOT extremes)
5. Volume confirmation: current volume > 0.8 * 20-period avg (liquidity filter)
6. ATR(14) 2.5x trailing stop for risk management
7. Position size: 0.20 (smaller for 15m frequency to reduce fee impact)

Why this should work on 15m:
- Session filter reduces trades by ~60% (only 12h of 24h)
- HTF alignment prevents counter-trend trades
- RSI pullback (not extreme) generates more entries than CRSI<10
- Volume filter avoids low-liquidity fake breakouts
- Target: 50-80 trades/year (within 40-100 target for 15m)

Entry conditions (LOOSE ENOUGH TO GENERATE TRADES):
- LONG = 4h bull + 1d bull + RSI(7) 30-45 + volume OK + session OK
- SHORT = 4h bear + 1d bear + RSI(7) 55-70 + volume OK + session OK
- Relaxed RSI thresholds (30-45/55-70 instead of 20/80) for more trades

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_hma_rsi_pullback_4h1d_v2"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    # Convert to seconds, then to datetime
    ts_seconds = open_time / 1000.0
    # Use pandas to extract hour (UTC)
    dt = pd.to_datetime(ts_seconds, unit='s', utc=True)
    return dt.hour

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
    
    # 1d regime: close vs open (bull/bear bias)
    daily_bias_raw = np.sign(df_1d['close'].values - df_1d['open'].values)
    daily_bias_aligned = align_htf_to_ltf(prices, df_1d, daily_bias_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    # Extract session hours
    session_hours = np.array([get_session_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    SIZE = 0.20  # Smaller size for 15m frequency
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(daily_bias_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        in_session = (session_hours[i] >= 0) and (session_hours[i] < 12)
        
        if not in_session:
            # Outside session: flatten positions
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA + 1d regime) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = daily_bias_aligned[i] > 0.0
        htf_1d_bear = daily_bias_aligned[i] < 0.0
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === RSI PULLBACK (LOOSE THRESHOLDS FOR TRADES) ===
        # Long: RSI 30-45 (pullback in uptrend, not oversold extreme)
        # Short: RSI 55-70 (pullback in downtrend, not overbought extreme)
        rsi_long_pullback = (rsi_7[i] >= 30) and (rsi_7[i] <= 45)
        rsi_short_pullback = (rsi_7[i] >= 55) and (rsi_7[i] <= 70)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 1d bull + RSI pullback + volume OK
        if htf_4h_bull and htf_1d_bull and rsi_long_pullback and volume_ok:
            desired_signal = SIZE
        
        # SHORT: 4h bear + 1d bear + RSI pullback + volume OK
        elif htf_4h_bear and htf_1d_bear and rsi_short_pullback and volume_ok:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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