#!/usr/bin/env python3
"""
Experiment #1245: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Session Filter

Hypothesis: 1h strategies failed (#1235, #1240 Sharpe=0.000) due to either 0 trades or 
too many trades. Key insight: use 4h/1d for DIRECTION, 1h only for ENTRY TIMING.

NEW APPROACH:
1. 1d HMA = macro trend filter (only long if price > 1d HMA, only short if <)
2. 4h ADX = regime filter (only trade if 4h ADX > 20 = trending, avoid chop)
3. 1h Fisher Transform = entry trigger (better than RSI for reversals, period=9)
4. Session filter = 8-20 UTC only (highest volume, avoid Asian session noise)
5. Volume filter = volume > 0.8x 20-bar avg (confirm participation)
6. Position size = 0.25 (smaller for 1h to reduce fee impact)

Target: 30-60 trades/year, Sharpe > 0.612 (beat current best), DD < -30%
Timeframe: 1h (use 4h/1d HTF via mtf_data helper)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_4h1d_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
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
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_fisher(close, period=9):
    """Ehlers Fisher Transform - catches reversals in bear/range markets"""
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period * 2:
        return fisher, trigger
    
    for i in range(period, n):
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        value = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * (0 if i == 0 else value)
        value = max(-0.999, min(0.999, value))
        
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        if i >= 1 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    vol_sma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

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
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h ADX for regime filter
    adx_4h_raw = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 1h to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Hysteresis: require signal change to persist for 2 bars
    signal_buffer = 0
    last_signal = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME FILTER (4h ADX) ===
        regime_trending = adx_4h_aligned[i] > 20.0
        
        # === FISHER TRANSFORM ENTRY ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # === ENTRY CONDITIONS (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: Macro bull + regime trending + session + volume + Fisher trigger
        if macro_bull and regime_trending and in_session and volume_ok and fisher_long:
            desired_signal = BASE_SIZE
        
        # SHORT: Macro bear + regime trending + session + volume + Fisher trigger
        elif macro_bear and regime_trending and in_session and volume_ok and fisher_short:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HYSTERESIS (prevent rapid flipping) ===
        if desired_signal != last_signal:
            signal_buffer += 1
            if signal_buffer >= 2:
                last_signal = desired_signal
                signal_buffer = 0
        else:
            signal_buffer = 0
        
        final_signal = last_signal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if final_signal > 0:
            final_signal = BASE_SIZE
        elif final_signal < 0:
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