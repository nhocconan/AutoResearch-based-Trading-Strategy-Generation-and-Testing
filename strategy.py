#!/usr/bin/env python3
"""
Experiment #1520: 1h Primary + 4h/12h HTF — Simplified HMA Trend + RSI Pullback + Volume Filter

Hypothesis: Previous 1h strategies failed due to OVER-FILTERING (session + volume + multiple confluence = 0 trades).
Key insights from failures #1510, #1515, #1518:
1. Session filters (8-20 UTC) KILL trade frequency on 1h — REMOVE session filter
2. Volume filter too strict (>0.8x) = no trades — use loose filter (>0.6x avg)
3. RSI bands too tight = no entries — use loose bands (35-65, not 40-60)
4. 4h HMA trend + 1h RSI pullback is proven pattern (#1513 worked on 1d)
5. Target: 40-80 trades/year on 1h (NOT >100 which causes fee drag)

Design:
- 4h HMA(21) for trend direction (HTF filter — slower, more reliable)
- 12h HMA(48) for macro bias (secondary HTF confirmation)
- 1h RSI(14) for pullback entries (loose: 35-65 range ensures trades happen)
- Volume filter: >0.6x 20-bar avg (loose, just filters extreme low volume)
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.25 (smaller for 1h to control fee impact)
- Target: 40-80 trades/train (4 years), 10-20 trades/test (15 months)

Timeframe: 1h (as required by experiment)
HTF: 4h (primary trend), 12h (macro bias)
Position Size: 0.25 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_volume_4h12h_atr_v1"
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_avg
    ratio[vol_avg <= 1e-10] = np.nan
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_1h[i]):
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
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (12h HMA) - primary direction bias ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - confirmation ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (1h HMA) - entry confirmation ===
        trend_1h_bull = close[i] > hma_1h[i]
        trend_1h_bear = close[i] < hma_1h[i]
        
        # === RSI PULLBACK - LOOSE bands for MORE trades ===
        # Long: RSI pulled back but not oversold (35-55)
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        # Short: RSI rallied but not overbought (45-65)
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        
        # === VOLUME FILTER - LOOSE (>0.6x avg) ===
        volume_ok = vol_ratio[i] > 0.6
        
        # === DESIRED SIGNAL - SIMPLIFIED FOR 1h ===
        desired_signal = 0.0
        
        # LONG: 12h bull + 4h bull + 1h bull + RSI pullback + volume
        # Option 1: Strong trend (all 3 TF bull) + RSI pullback + volume
        if macro_bull and trend_4h_bull and trend_1h_bull and rsi_pullback_long and volume_ok:
            desired_signal = BASE_SIZE
        # Option 2: 12h + 4h bull + 1h bull + RSI pullback (slightly looser)
        elif macro_bull and trend_4h_bull and trend_1h_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE * 0.9
        # Option 3: 12h + 4h bull + 1h above HMA + RSI not overbought (loosest for trades)
        elif macro_bull and trend_4h_bull and trend_1h_bull and rsi[i] < 60.0:
            desired_signal = BASE_SIZE * 0.7
        # Option 4: 12h bull + 4h bull + RSI pullback (fallback)
        elif macro_bull and trend_4h_bull and rsi_pullback_long and volume_ok:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT: 12h bear + 4h bear + 1h bear + RSI pullback + volume
        # Option 1: Strong trend (all 3 TF bear) + RSI pullback + volume
        elif macro_bear and trend_4h_bear and trend_1h_bear and rsi_pullback_short and volume_ok:
            desired_signal = -BASE_SIZE
        # Option 2: 12h + 4h bear + 1h bear + RSI pullback (slightly looser)
        elif macro_bear and trend_4h_bear and trend_1h_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE * 0.9
        # Option 3: 12h + 4h bear + 1h below HMA + RSI not oversold (loosest for trades)
        elif macro_bear and trend_4h_bear and trend_1h_bear and rsi[i] > 40.0:
            desired_signal = -BASE_SIZE * 0.7
        # Option 4: 12h bear + 4h bear + RSI pullback (fallback)
        elif macro_bear and trend_4h_bear and rsi_pullback_short and volume_ok:
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.4:
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