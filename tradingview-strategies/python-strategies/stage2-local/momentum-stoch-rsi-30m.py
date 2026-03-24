#!/usr/bin/env python3

name = "Momentum Strategy (BTC/USDT; 30m) - STOCH RSI"
timeframe = "30m"
leverage = 1

import numpy as np
import pandas as pd


def _rsi(close, length):
    """Calculate RSI using pandas/numpy only."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initial SMA for first 'length' periods
    if length <= len(close):
        avg_gain[length - 1] = np.mean(gain[:length])
        avg_loss[length - 1] = np.mean(loss[:length])
        
        # EMA smoothing for rest
        for i in range(length, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i - 1] * (length - 1) + loss[i]) / length
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _stoch_rsi(rsi_values, length):
    """Calculate Stochastic of RSI."""
    stoch = np.zeros_like(rsi_values)
    
    for i in range(length - 1, len(rsi_values)):
        lowest = np.min(rsi_values[i - length + 1:i + 1])
        highest = np.max(rsi_values[i - length + 1:i + 1])
        if highest != lowest:
            stoch[i] = 100 * (rsi_values[i] - lowest) / (highest - lowest)
        else:
            stoch[i] = 50.0
    
    return stoch


def _ema(series, length):
    """Calculate EMA using numpy."""
    ema = np.zeros_like(series)
    multiplier = 2 / (length + 1)
    
    if len(series) > 0:
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = (series[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def _crossover(series1, series2):
    """Detect crossover (series1 crosses above series2)."""
    crossover = np.zeros(len(series1), dtype=bool)
    for i in range(1, len(series1)):
        if series1[i - 1] <= series2[i - 1] and series1[i] > series2[i]:
            crossover[i] = True
    return crossover


def _crossunder(series1, series2):
    """Detect crossunder (series1 crosses below series2)."""
    crossunder = np.zeros(len(series1), dtype=bool)
    for i in range(1, len(series1)):
        if series1[i - 1] >= series2[i - 1] and series1[i] < series2[i]:
            crossunder[i] = True
    return crossunder


def _barssince(condition):
    """Calculate bars since condition was true."""
    result = np.full(len(condition), np.inf)
    last_true = -np.inf
    
    for i in range(len(condition)):
        if condition[i]:
            last_true = i
        result[i] = i - last_true if last_true != -np.inf else np.inf
    
    return result


def generate_signals(prices):
    """
    Generate target position signals based on Stochastic RSI momentum strategy.
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume]
    
    Returns:
        numpy.ndarray with target position fractions (-1, 0, or 1) for each bar.
    """
    n = len(prices)
    if n == 0:
        return np.array([])
    
    close = prices['close'].values.astype(float)
    high = prices['high'].values.astype(float)
    low = prices['low'].values.astype(float)
    
    # Strategy parameters (from Pine Script defaults)
    smoothK = 3
    smoothD = 6
    lengthRSI = 12
    lengthStoch = 12
    ipercomprato = 85.29  # Overbought level
    ipervenduto = 30.6    # Oversold level
    BarsDelay = 6
    periodomediamobile_fast = 1
    periodomediamobile_slow = 60
    
    # Stop loss and take profit percentages
    sl_long_pct = 10.0 / 100.0
    tp_long_pct = 8.0 / 100.0
    sl_short_pct = 20.0 / 100.0
    tp_short_pct = 35.0 / 100.0
    
    # Gamble sizing parameters
    use_gamble = True
    delta_size = 25.0
    size_limit = 100.0
    size_init = 50.0
    
    # Calculate indicators
    rsi1 = _rsi(close, lengthRSI)
    stoch_raw = _stoch_rsi(rsi1, lengthStoch)
    k = _ema(stoch_raw, smoothK)
    d = _ema(k, smoothD)
    
    # EMA trend filter
    EMAfast = _ema(close, periodomediamobile_fast)
    EMAslow = _ema(close, periodomediamobile_slow)
    
    # Cross signals
    goldencross = _crossover(k, d)
    deathcross = _crossunder(k, d)
    
    # Track value at cross for entry conditions
    valoreoro = np.zeros(n)
    valoremorte = np.zeros(n)
    
    last_golden_val = np.nan
    last_death_val = np.nan
    for i in range(n):
        if goldencross[i]:
            last_golden_val = d[i]
        if deathcross[i]:
            last_death_val = d[i]
        valoreoro[i] = last_golden_val if not np.isnan(last_golden_val) else d[i]
        valoremorte[i] = last_death_val if not np.isnan(last_death_val) else d[i]
    
    # Entry conditions
    siamoinipervenduto = goldencross & (valoreoro <= ipervenduto)
    siamoinipercomprato = deathcross & (valoremorte >= ipercomprato)
    
    # Trend filters
    siamoinuptrend = EMAfast > EMAslow
    siamoindowntrend = EMAfast < EMAslow
    
    # Long entry condition (with trend filter)
    CondizioneAperturaLong = siamoinipervenduto & siamoinuptrend
    CondizioneChiusuraLong = siamoinipercomprato
    
    # Short entry condition (with trend filter)
    CondizioneAperturaShort = siamoinipercomprato & siamoindowntrend
    CondizioneChiusuraShort = siamoinipervenduto
    
    # Bars delay for exits
    sonPassateLeBarreD = _barssince(CondizioneChiusuraLong) == BarsDelay
    sonPassateLeBarreDs = _barssince(CondizioneChiusuraShort) == BarsDelay
    
    # Initialize signals array
    signals = np.zeros(n)
    
    # State tracking for position management
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    sizeordine = size_init
    last_trade_result = 0  # 0 = none, 1 = win, -1 = loss
    
    # Track stop loss and take profit levels
    sl_long_level = 0.0
    tp_long_level = 0.0
    sl_short_level = 0.0
    tp_short_level = 0.0
    
    for i in range(1, n):
        prev_position = position
        
        # Check stop loss and take profit hits using bar high/low (approximation)
        if position == 1:  # Long position
            if low[i] <= sl_long_level or high[i] >= tp_long_level:
                # Exit due to SL/TP hit
                position = 0
                # Track trade result for gamble sizing
                if low[i] <= sl_long_level:
                    last_trade_result = -1
                else:
                    last_trade_result = 1
                # Adjust size for gamble sizing
                if use_gamble:
                    if last_trade_result == -1 and sizeordine < size_limit:
                        sizeordine = min(sizeordine + delta_size, size_limit)
                    elif last_trade_result == 1:
                        sizeordine = size_init
        
        elif position == -1:  # Short position
            if high[i] >= sl_short_level or low[i] <= tp_short_level:
                # Exit due to SL/TP hit
                position = 0
                # Track trade result for gamble sizing
                if high[i] >= sl_short_level:
                    last_trade_result = -1
                else:
                    last_trade_result = 1
                # Adjust size for gamble sizing
                if use_gamble:
                    if last_trade_result == -1 and sizeordine < size_limit:
                        sizeordine = min(sizeordine + delta_size, size_limit)
                    elif last_trade_result == 1:
                        sizeordine = size_init
        
        # Check signal-based exits (with bars delay)
        if position == 1 and CondizioneChiusuraLong[i]:
            if siamoinuptrend[i] and sonPassateLeBarreD[i]:
                position = 0
                last_trade_result = 0  # Signal exit, not SL/TP
            elif not siamoinuptrend[i]:
                position = 0
                last_trade_result = 0
        
        if position == -1 and CondizioneChiusuraShort[i]:
            if siamoindowntrend[i] and sonPassateLeBarreDs[i]:
                position = 0
                last_trade_result = 0
            elif not siamoindowntrend[i]:
                position = 0
                last_trade_result = 0
        
        # Check entries (only if flat)
        if position == 0:
            if CondizioneAperturaLong[i]:
                position = 1
                entry_price = close[i - 1]  # Previous close for next-bar execution
                sl_long_level = entry_price * (1 - sl_long_pct)
                tp_long_level = entry_price * (1 + tp_long_pct)
            elif CondizioneAperturaShort[i]:
                position = -1
                entry_price = close[i - 1]
                sl_short_level = entry_price * (1 + sl_short_pct)
                tp_short_level = entry_price * (1 - tp_short_pct)
        
        # Set signal (target position)
        signals[i] = position
    
    return signals
