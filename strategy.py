#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + Weekly Trend + Volume

HYPOTHESIS: Donchian channel breakouts capture momentum moves. Weekly HMA provides 
trend bias to avoid counter-trend trades. Volume confirmation ensures institutional 
participation. This works in BOTH bull and bear markets because we trade breakouts 
in the direction of the higher timeframe trend.

WHY THIS SHOULD WORK:
- Bull market: Weekly HMA up + Donchian upper breakout = long momentum
- Bear market: Weekly HMA down + Donchian lower breakout = short momentum
- 12h timeframe: Fewer trades than 4h, less fee drag, still enough signals

TARGET: 75-200 total trades over 4 years (19-50/year for 12h)
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95tr)

KEY DESIGN:
1. Weekly HMA(21) for trend bias (loose filter - just direction)
2. Donchian(20) breakout on 12h (proven pattern)
3. Volume > 1.3x 20-avg (confirmation, not too strict)
4. ATR(14) stoploss at 2.5x
5. Signal: 0.30 (discrete, minimizes churn)
6. LOOSE entry conditions to ensure >=50 trades total
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_weekly_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper and lower bounds"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA for trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for additional confirmation
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === TREND BIAS (Weekly HMA) ===
        # Simple: price above weekly HMA = bullish bias, below = bearish
        price_above_1w_hma = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        
        # === VOLUME CONFIRMATION (loose threshold) ===
        vol_ok = vol_ratio[i] > 1.2  # Just above average, not strict
        
        # === DONCHIAN BREAKOUT ===
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Upper breakout (price crosses above Donchian upper)
        upper_breakout = high[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        
        # Lower breakout (price crosses below Donchian lower)
        lower_breakout = low[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: Upper breakout + bullish bias OR just strong breakout
        if upper_breakout:
            if price_above_1w_hma and vol_ok:
                desired_signal = SIZE
            elif price_above_1w_hma:
                # Even without volume, enter if trend aligns
                desired_signal = SIZE
            elif vol_ratio[i] > 1.5:
                # Strong volume alone can trigger
                desired_signal = SIZE
        
        # SHORT: Lower breakout + bearish bias OR just strong breakout
        if lower_breakout:
            if not price_above_1w_hma and vol_ok:
                desired_signal = -SIZE
            elif not price_above_1w_hma:
                # Even without volume, enter if trend aligns
                desired_signal = -SIZE
            elif vol_ratio[i] > 1.5:
                # Strong volume alone can trigger
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
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
        
        # === TAKE PROFIT (opposite Donchian band) ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at lower band (mean reversion)
            if not np.isnan(donchian_lower[i]) and low[i] <= donchian_lower[i] * 1.02:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at upper band (mean reversion)
            if not np.isnan(donchian_upper[i]) and high[i] >= donchian_upper[i] * 0.98:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals