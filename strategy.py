#!/usr/bin/env python3
"""
Experiment #1515: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: Based on #1513 success (1d HMA+RSI+Donchian), scaling to 1h with 4h/1d HTF
should work IF we add strict confluence filters to limit trade frequency.
Key insights from 1125+ failed strategies:
1. Lower TF (1h) needs STRICTER entries to avoid fee drag (>100 trades/yr = death)
2. HTF (4h/1d) for SIGNAL DIRECTION, 1h only for ENTRY TIMING
3. Session filter (8-20 UTC) reduces Asian session noise
4. Volume filter ensures real moves, not fakeouts
5. 3+ confluence required: HTF trend + RSI pullback + volume + session

Design:
- 1d HMA(21) for macro trend bias (strongest HTF filter)
- 4h HMA(21) for secondary trend confirmation
- 1h RSI(14) for pullback entries (strict: 25-45 long, 55-75 short)
- Volume > 0.8x 20-period average (confirms real moves)
- Session filter: only 8-20 UTC (London/NY overlap, avoids Asian noise)
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.25 (smaller than 1d due to higher trade frequency)
- Target: 40-80 trades/train (4 years), 10-20 trades/test (15 months)

Timeframe: 1h (as required by experiment #1515)
HTF: 4h and 1d (dual HTF for stronger trend confirmation)
Position Size: 0.25 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h1d_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) // 3600) % 24
    return hours

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
    
    # Calculate and align 4h HMA for trend confirmation
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend bias (strongest filter)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Extract UTC hour for session filter
    utc_hour = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h (40-80 trades/year target)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
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
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) - avoids Asian session noise ===
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === VOLUME FILTER - confirms real moves ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO TREND (1d HMA) - strongest direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) - confirmation ===
        fourh_bull = close[i] > hma_4h_aligned[i]
        fourh_bear = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK - STRICT bands for FEWER trades ===
        # Long: RSI pulled back to oversold zone (25-45)
        rsi_pullback_long = 25.0 <= rsi[i] <= 45.0
        # Short: RSI rallied to overbought zone (55-75)
        rsi_pullback_short = 55.0 <= rsi[i] <= 75.0
        
        # === DESIRED SIGNAL - STRICT 4+ CONFLUENCE FOR 1h ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 4h bull + RSI pullback + volume + session (ALL 5 required)
        if daily_bull and fourh_bull and rsi_pullback_long and volume_confirmed and in_session:
            desired_signal = BASE_SIZE
        # LONG fallback: 1d bull + 4h bull + RSI pullback + volume (4/5, skip session)
        elif daily_bull and fourh_bull and rsi_pullback_long and volume_confirmed:
            desired_signal = BASE_SIZE * 0.8
        # LONG loosest: 1d bull + 4h bull + RSI not overbought + volume (ensures some trades)
        elif daily_bull and fourh_bull and rsi[i] < 55.0 and volume_confirmed:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT: 1d bear + 4h bear + RSI pullback + volume + session (ALL 5 required)
        elif daily_bear and fourh_bear and rsi_pullback_short and volume_confirmed and in_session:
            desired_signal = -BASE_SIZE
        # SHORT fallback: 1d bear + 4h bear + RSI pullback + volume (4/5, skip session)
        elif daily_bear and fourh_bear and rsi_pullback_short and volume_confirmed:
            desired_signal = -BASE_SIZE * 0.8
        # SHORT loosest: 1d bear + 4h bear + RSI not oversold + volume (ensures some trades)
        elif daily_bear and fourh_bear and rsi[i] > 45.0 and volume_confirmed:
            desired_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals