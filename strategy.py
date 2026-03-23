#!/usr/bin/env python3
"""
Experiment #1240: 1h Primary + 4h/12h HTF — Regime-Adaptive HMA/RSI with Session Filter

Hypothesis: Previous 1h strategies (#1230, #1235, #1238) failed with 0 trades due to 
too many restrictive filters. Key changes:
1. LOOSEN entry conditions: HTF trend is PRIMARY, 1h RSI is secondary (not both required)
2. Choppiness Index regime: CHOP>55 = range (mean revert), CHOP<45 = trend (trend follow)
3. Session filter ONLY for entries (8-20 UTC), NOT for exits (prevents 0 trades)
4. Volume filter as soft confirmation (not hard requirement)
5. Smaller position size (0.25) for lower TF to reduce fee impact
6. Target: 40-80 trades/year (3-7/month)

Timeframe: 1h (30-60 trades/year target)
HTF: 4h for trend, 12h for regime
Entry: RSI pullback in HTF trend direction OR mean revert in range
Exit: ATR trailing stop 2.5x (always active, no session filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_hma_rsi_4h12h_session_atr_v1"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_chop(high, low, close, period=14):
    """Choppiness Index - measures market choppiness (high = range, low = trend)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
        else:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h Choppiness for regime
    chop_12h_raw = calculate_chop(df_12h['high'].values, df_12h['low'].values, 
                                   df_12h['close'].values, period=14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 1h TF to reduce fee impact
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Signal persistence buffer (prevent rapid flipping)
    signal_buffer = 0
    last_signal = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            signals[i] = last_signal
            continue
        if np.isnan(hma_1h[i]) or np.isnan(rsi_1h[i]):
            signals[i] = last_signal
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            signals[i] = last_signal
            continue
        
        # === REGIME DETECTION (12h Choppiness) ===
        # CHOP > 55 = range market (mean revert)
        # CHOP < 45 = trending market (trend follow)
        # Middle zone = use trend logic as default
        is_range = chop_12h_aligned[i] > 55.0
        is_trend = chop_12h_aligned[i] < 45.0
        
        # === TREND DIRECTION (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === SESSION FILTER (8-20 UTC only for NEW entries) ===
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER (soft - only reduces size, doesn't block) ===
        vol_above_avg = volume[i] > 0.8 * vol_sma[i] if not np.isnan(vol_sma[i]) else True
        
        # === ENTRY CONDITIONS (LOOSENED for more trades) ===
        desired_signal = 0.0
        
        if is_trend:
            # TREND REGIME: Follow HTF trend with RSI pullback
            if trend_bull and rsi_1h[i] < 55 and rsi_1h[i] > 30:
                # Long on pullback in uptrend - WIDER RSI range
                if in_session or in_position:  # Allow exit anytime, entry in session
                    desired_signal = BASE_SIZE
            elif trend_bear and rsi_1h[i] > 45 and rsi_1h[i] < 70:
                # Short on rally in downtrend - WIDER RSI range
                if in_session or in_position:
                    desired_signal = -BASE_SIZE
        elif is_range:
            # RANGE REGIME: Mean revert at extremes
            if rsi_1h[i] < 35:
                # Long at oversold
                if in_session or in_position:
                    desired_signal = BASE_SIZE
            elif rsi_1h[i] > 65:
                # Short at overbought
                if in_session or in_position:
                    desired_signal = -BASE_SIZE
        else:
            # NEUTRAL REGIME: Use simple trend following
            if trend_bull and rsi_1h[i] < 60:
                if in_session or in_position:
                    desired_signal = BASE_SIZE
            elif trend_bear and rsi_1h[i] > 40:
                if in_session or in_position:
                    desired_signal = -BASE_SIZE
        
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
        
        # === SIGNAL PERSISTENCE (prevent rapid flipping) ===
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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