#!/usr/bin/env python3
"""
Experiment #1470: 1h Primary + 4h/12h HTF — Fisher Transform Regime with Volume Session Filter

Hypothesis: Previous 1h strategies failed due to OVER-FILTERING (0 trades). This version uses:
1. Fisher Transform (period=9) - naturally generates fewer signals than RSI, catches reversals
2. 12h HMA(21) - macro trend direction (proven from best strategies)
3. 4h ADX(14) - regime filter (ADX>25=trend, ADX<20=range)
4. Volume filter - only trade when volume > 1.2x 20-bar average
5. Session filter - only 8-20 UTC (London/NY overlap = highest liquidity)
6. Fisher extremes at -1.5/+1.5 (wider than typical -1.0/+1.0 for fewer signals)

Why this should work for 1h:
- Fisher Transform normalizes price into bounded range, generates cleaner signals than RSI
- 12h HMA provides direction bias without over-filtering
- 4h ADX adds regime context (trend vs mean-revert)
- Volume + session filters naturally limit trades to 30-80/year target
- Position size 0.25 (smaller for 1h to reduce fee drag)
- 2.5x ATR trailing stop (tighter than 12h strategies due to lower TF)

Target: 30-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_regime_4h12h_hma_adx_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price into bounded range (-1 to +1)
    Catches reversals better than RSI in bear/range markets
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    # Calculate typical price and normalized price
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 0.999 * (2.0 * (close[i] - lowest) / (highest - lowest) - 1.0)
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    
    # Initial values
    plus_di[period-1] = 100 * np.sum(plus_dm[:period]) / np.sum(tr[:period]) if np.sum(tr[:period]) > 0 else 0
    minus_di[period-1] = 100 * np.sum(minus_dm[:period]) / np.sum(tr[:period]) if np.sum(tr[:period]) > 0 else 0
    atr[period-1] = np.sum(tr[:period]) / period
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * (plus_dm[i] + (period - 1) * (plus_di[i-1] / 100 * atr[i-1] / 100 * period)) / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * (minus_dm[i] + (period - 1) * (minus_di[i-1] / 100 * atr[i-1] / 100 * period)) / atr[i] if atr[i] > 0 else 0
    
    # Calculate ADX
    adx = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(period * 2 - 1, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period:period*2])
    for i in range(period * 2, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

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

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from timestamp (milliseconds)"""
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h ADX for regime
    adx_4h_raw = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h to reduce fee drag
    
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
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(adx_4h_aligned[i]):
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
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_timestamp(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (must be > 1.2x average) ===
        volume_ok = volume[i] > 1.2 * vol_sma[i]
        
        # === MACRO TREND (12h HMA) - direction filter ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === REGIME (4h ADX) - trend vs range ===
        adx_value = adx_4h_aligned[i]
        is_trending = adx_value > 25.0
        is_ranging = adx_value < 20.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_extreme_low = fisher[i] < -1.5
        fisher_extreme_high = fisher[i] > 1.5
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES (require: session + volume + macro bull)
        if in_session and volume_ok and macro_bull:
            # Path 1: Fisher crosses up from extreme low in trending regime
            if fisher_cross_up and is_trending:
                desired_signal = BASE_SIZE
            # Path 2: Fisher at extreme low in ranging regime (mean reversion)
            elif fisher_extreme_low and is_ranging:
                desired_signal = BASE_SIZE
            # Path 3: Fisher rising from deep oversold + macro bull
            elif fisher[i] < -1.0 and fisher[i] > fisher_signal[i] and macro_bull:
                desired_signal = BASE_SIZE * 0.7
        
        # SHORT ENTRIES (require: session + volume + macro bear)
        elif in_session and volume_ok and macro_bear:
            # Path 1: Fisher crosses down from extreme high in trending regime
            if fisher_cross_down and is_trending:
                desired_signal = -BASE_SIZE
            # Path 2: Fisher at extreme high in ranging regime (mean reversion)
            elif fisher_extreme_high and is_ranging:
                desired_signal = -BASE_SIZE
            # Path 3: Fisher falling from deep overbought + macro bear
            elif fisher[i] > 1.0 and fisher[i] < fisher_signal[i] and macro_bear:
                desired_signal = -BASE_SIZE * 0.7
        
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
        if desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE
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