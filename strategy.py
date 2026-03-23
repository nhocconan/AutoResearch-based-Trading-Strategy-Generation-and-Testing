#!/usr/bin/env python3
"""
Experiment #1402: 12h Primary + 1d/1w HTF — Triple HMA Trend + Donchian Breakout + Volume

Hypothesis: Previous 12h strategy (#1396 Sharpe=0.525) was close to 1d best (0.618) but lacked
ultra-macro filter. Adding 1w HMA as strongest trend filter should improve win rate on major
moves while 1d HMA provides intermediate confirmation. Volume confirmation on breakouts
filters false breakouts (proven in 4h strategies).

Key improvements over #1396:
1. Add 1w HMA(21) as ultra-macro filter (strongest trend confirmation)
2. Volume confirmation on breakouts (volume > 1.5x 20-period average)
3. Adaptive position sizing: 0.30 when 1w+1d agree, 0.20 when only 1d confirms
4. Asymmetric stoploss: 2.0x ATR in strong trend, 3.0x otherwise
5. RSI momentum threshold tightened (40-60 vs 30-70) for better entry quality

Design:
1. 1w HMA(21) = ultra-macro trend (strongest filter, only trade with weekly trend)
2. 1d HMA(21) = intermediate trend confirmation
3. 12h Donchian(20/55) = entry triggers with volume confirmation
4. RSI(14) = momentum filter (40-60 range for quality entries)
5. ATR(14) trailing stop = adaptive (2.0x strong trend, 3.0x weak)
6. Position size: 0.30 (1w+1d agree), 0.20 (1d only), 0.0 (against 1w)

Target: 25-45 trades/year, Sharpe > 0.618 (beat 1d baseline), trades >= 30 train, >= 5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_triple_hma_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum confirmation"""
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
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_sma[i] = np.nanmean(volume[i-period+1:i+1])
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    donchian_55_upper, donchian_55_lower = calculate_donchian(high, low, period=55)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE_STRONG = 0.30  # When 1w + 1d agree
    BASE_SIZE_WEAK = 0.20    # When only 1d confirms
    
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
        if np.isnan(donchian_20_upper[i]) or np.isnan(donchian_55_upper[i]):
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
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
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
        
        # === ULTRA-MACRO TREND (1w HMA) - strongest filter ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) - confirmation ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (both agree = strong) ===
        strong_bull = weekly_bull and daily_bull
        strong_bear = weekly_bear and daily_bear
        weak_bull = daily_bull and not weekly_bull
        weak_bear = daily_bear and not weekly_bear
        
        # === RSI MOMENTUM (tighter bands for quality) ===
        rsi_bull = rsi[i] > 40.0
        rsi_bear = rsi[i] < 60.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === VOLUME CONFIRMATION (breakout validity) ===
        volume_confirmed = volume[i] > 1.5 * vol_sma[i]
        
        # === DUAL DONCHIAN BREAKOUT ===
        breakout_20_long = close[i] > donchian_20_upper[i-1]
        breakout_20_short = close[i] < donchian_20_lower[i-1]
        breakout_55_long = close[i] > donchian_55_upper[i-1]
        breakout_55_short = close[i] < donchian_55_lower[i-1]
        
        # === DESIRED SIGNAL - MULTIPLE ENTRY PATHS ===
        desired_signal = 0.0
        use_strong_size = False
        
        # LONG ENTRY PATHS
        # Path 1: Strong trend + Donchian-20 + volume (best setup)
        if strong_bull and breakout_20_long and volume_confirmed and rsi_bull:
            desired_signal = BASE_SIZE_STRONG
            use_strong_size = True
        # Path 2: Strong trend + Donchian-55 + RSI strong (major breakout)
        elif strong_bull and breakout_55_long and rsi_strong_bull:
            desired_signal = BASE_SIZE_STRONG
            use_strong_size = True
        # Path 3: Weak bull + Donchian-20 + volume (secondary entry)
        elif weak_bull and breakout_20_long and volume_confirmed:
            desired_signal = BASE_SIZE_WEAK
        # Path 4: Daily bull + RSI momentum (trend continuation)
        elif daily_bull and rsi[i] > 45.0 and not weekly_bear:
            desired_signal = BASE_SIZE_WEAK * 0.5
        
        # SHORT ENTRY PATHS
        # Path 1: Strong trend + Donchian-20 + volume (best setup)
        elif strong_bear and breakout_20_short and volume_confirmed and rsi_bear:
            desired_signal = -BASE_SIZE_STRONG
            use_strong_size = True
        # Path 2: Strong trend + Donchian-55 + RSI strong (major breakout)
        elif strong_bear and breakout_55_short and rsi_strong_bear:
            desired_signal = -BASE_SIZE_STRONG
            use_strong_size = True
        # Path 3: Weak bear + Donchian-20 + volume (secondary entry)
        elif weak_bear and breakout_20_short and volume_confirmed:
            desired_signal = -BASE_SIZE_WEAK
        # Path 4: Daily bear + RSI momentum (trend continuation)
        elif daily_bear and rsi[i] < 55.0 and not weekly_bull:
            desired_signal = -BASE_SIZE_WEAK * 0.5
        
        # === ADAPTIVE STOPLOSS (2.0x strong trend, 3.0x weak) ===
        stoploss_multiplier = 2.0 if use_strong_size else 3.0
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - stoploss_multiplier * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + stoploss_multiplier * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if abs(desired_signal) >= BASE_SIZE_WEAK * 0.4:
            if desired_signal > 0:
                final_signal = BASE_SIZE_STRONG if use_strong_size else BASE_SIZE_WEAK
            else:
                final_signal = -BASE_SIZE_STRONG if use_strong_size else -BASE_SIZE_WEAK
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