#!/usr/bin/env python3
"""
Experiment #218: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume Filter

Hypothesis: After 12h/1d failures with complex regime switching (#206-217), return to 
proven 4h HMA trend foundation but execute on 30m for better entry timing. Key insight:
lower TF needs STRICT filters to avoid fee drag, but NOT so strict that 0 trades occur.

Strategy components:
1. 4h HMA(21) for macro trend direction (proven in current best strategy)
2. 1d HMA(21) for higher-timeframe bias filter
3. 30m RSI(7) for faster pullback detection (not extremes, 35-65 range)
4. Volume filter: current > 0.8x 20-period average
5. Session filter: only trade 8-20 UTC (high liquidity windows)
6. ATR(14) 2.5x trailing stoploss

Why this might work:
- 4h HMA trend is proven (current best uses it)
- 30m entries give better timing than 4h-only
- RSI(7) faster than RSI(14) for lower TF
- Volume + session filters reduce false breakouts
- Discrete position sizes (0.0, ±0.20, ±0.30) minimize fee churn

TARGET: 40-80 trades/year on 30m, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.20, ±0.30 (discrete to minimize fee churn)
Stoploss: ATR(14) 2.5x trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_vol_session_4h1d_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_sma(series, period):
    """Calculate Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

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
    
    # Calculate 30m indicators (primary timeframe)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # Calculate 4h HMA for trend (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Get UTC hour for session filter
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === HTF MACRO BIAS (4h + 1d HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND BIAS (4h HMA slope) ===
        hma_4h_bullish = price_above_hma_4h
        hma_4h_bearish = price_below_hma_4h
        
        # === RSI PULLBACK FILTER (not extremes) ===
        # For longs: RSI pulled back but not oversold (35-55)
        # For shorts: RSI rallied but not overbought (45-65)
        rsi_long_ok = 35.0 <= rsi_7[i] <= 55.0
        rsi_short_ok = 45.0 <= rsi_7[i] <= 65.0
        
        # === MOMENTUM CONFIRMATION ===
        # Price above both 4h and 1d HMA = strong bullish
        # Price below both 4h and 1d HMA = strong bearish
        strong_bullish = price_above_hma_4h and price_above_hma_1d
        strong_bearish = price_below_hma_4h and price_below_hma_1d
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + RSI pullback + volume + session
        if hma_4h_bullish and in_session and volume_ok:
            if rsi_long_ok:
                if strong_bullish:
                    new_signal = POSITION_SIZE_FULL  # With both HTF trends
                else:
                    new_signal = POSITION_SIZE_HALF  # Only 4h bullish
            elif rsi_7[i] < 40.0 and strong_bullish:
                # Deeper pullback in strong uptrend
                new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: 4h bearish + RSI pullback + volume + session
        elif hma_4h_bearish and in_session and volume_ok:
            if rsi_short_ok:
                if strong_bearish:
                    new_signal = -POSITION_SIZE_FULL  # With both HTF trends
                else:
                    new_signal = -POSITION_SIZE_HALF  # Only 4h bearish
            elif rsi_7[i] > 60.0 and strong_bearish:
                # Stronger rally in strong downtrend
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid (relax entry filters for holding)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h HMA still bullish (relaxed RSI)
                if hma_4h_bullish and rsi_7[i] < 70.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h HMA still bearish (relaxed RSI)
                if hma_4h_bearish and rsi_7[i] > 30.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h HMA turns bearish
        if in_position and position_side > 0 and hma_4h_bearish:
            new_signal = 0.0
        
        # Exit short if 4h HMA turns bullish
        if in_position and position_side < 0 and hma_4h_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals