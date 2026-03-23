#!/usr/bin/env python3
"""
Experiment #1410: 1h Primary + 4h/12h HTF — Regime-Adaptive HMA + Fisher Transform

Hypothesis: 1h timeframe with dual HTF trend filtering + Fisher Transform reversals
will work better than pure trend-following on lower TFs. Key insights from failures:
- #1405 (CHOP+CRSI+session+volume) had Sharpe=-2.3 due to TOO MANY filters = 0 trades
- #1400 (1h Donchian+HMA) had Sharpe=0.049 — too simple, no regime adaptation
- #1402 (12h Triple HMA) had Sharpe=0.545 — HTF trend works well

Design:
1. 12h HMA(21) = macro trend direction (ONLY trade WITH this trend)
2. 4h HMA(21) = intermediate confirmation (must agree with 12h)
3. 1h Fisher Transform(9) = entry timing (catches reversals in bear rallies)
4. 1h RSI(14) = momentum confirmation (wide bands 35-65 for trade frequency)
5. 1h ATR(14) trailing stop 2.5x = risk management
6. Session filter (8-20 UTC) = avoid low liquidity whipsaws
7. Volume filter (>0.7x 20-bar avg) = ensure real moves

Position size: 0.25 (conservative for 1h volatility)
Target: 40-80 trades/year, Sharpe > 0.618 (beat 1d baseline)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_fisher_regime_4h12h_atr_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals sharply, excellent for bear market rallies
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        if highest > lowest:
            normalized = (hl2 - lowest) / (highest - lowest)
            # Clamp to avoid division errors
            normalized = max(0.001, min(0.999, normalized))
            
            # Apply Fisher transform
            fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            # Smooth with previous value (EMA-like)
            if i > period - 1 and not np.isnan(fisher[i-1]):
                fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            else:
                fisher[i] = fisher_val
            
            # Trigger line (previous fisher)
            if i > period - 1:
                trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_rsi(close, period=14):
    """Relative Strength Index - wide bands for entry confirmation"""
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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = np.full(n, np.nan)
    for i in range(period - 1, n):
        vol_avg[i] = np.nanmean(volume[i-period+1:i+1])
    
    return vol_avg

def extract_hour(open_time):
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (12h HMA) - strongest filter ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - confirmation ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === TREND CONFLUENCE (both HTF must agree) ===
        trend_bull = macro_bull and inter_bull
        trend_bear = macro_bear and inter_bear
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # === RSI MOMENTUM (WIDE bands to ensure trades) ===
        rsi_bull = rsi[i] > 35.0
        rsi_bear = rsi[i] < 65.0
        rsi_strong_bull = rsi[i] > 45.0
        rsi_strong_bear = rsi[i] < 55.0
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = extract_hour(open_time[i])
        session_ok = 8 <= hour <= 20
        
        # === DESIRED SIGNAL - REGIME ADAPTIVE ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bull trend + Fisher reversal OR RSI momentum + volume + session
        if trend_bull and volume_ok and session_ok:
            # Path 1: Fisher reversal (strongest signal)
            if fisher_long:
                desired_signal = BASE_SIZE
            # Path 2: RSI momentum pullback (moderate signal)
            elif rsi_strong_bull and close[i] > hma_4h_aligned[i]:
                desired_signal = BASE_SIZE * 0.6
            # Path 3: RSI neutral + trend continuation (weaker signal)
            elif rsi_bull and close[i] > hma_12h_aligned[i]:
                desired_signal = BASE_SIZE * 0.4
        
        # SHORT ENTRY: HTF bear trend + Fisher reversal OR RSI momentum + volume + session
        elif trend_bear and volume_ok and session_ok:
            # Path 1: Fisher reversal (strongest signal)
            if fisher_short:
                desired_signal = -BASE_SIZE
            # Path 2: RSI momentum pullback (moderate signal)
            elif rsi_strong_bear and close[i] < hma_4h_aligned[i]:
                desired_signal = -BASE_SIZE * 0.6
            # Path 3: RSI neutral + trend continuation (weaker signal)
            elif rsi_bear and close[i] < hma_12h_aligned[i]:
                desired_signal = -BASE_SIZE * 0.4
        
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
        if desired_signal >= BASE_SIZE * 0.35:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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