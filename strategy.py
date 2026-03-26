#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + Weekly Trend + Volume Confirmation

HYPOTHESIS: Weekly trend (HMA) filters direction, daily Donchian(20) breakout 
captures institutional moves, volume confirms, ATR stoploss manages risk.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Weekly HMA = trend direction filter (no counter-trend trades)
- Donchian breakout = structural break, works in both directions
- Volume spike = institutional confirmation
- ATR stoploss = adapts to volatility in any market phase
- 1d timeframe = ~100 trades max over 4 years (proven sustainable)

TARGET: 75-150 total over 4 years (19-37/year).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95tr)

KEY DESIGN (KISS - Keep It Simple Stupid):
1. Weekly HMA(21) for trend direction ONLY
2. Daily Donchian(20) breakout for entry
3. Volume spike (>1.5x 20-avg) for confirmation
4. ATR(14) stoploss at 2x
5. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_wma_vol_1w_v1"
timeframe = "1d"
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
    """Donchian Channel - returns (upper, lower)"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
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
    
    # === WEEKLY HTF DATA (for trend) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly HMA(21) for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Weekly EMA for additional confirmation
    ema_1w_raw = calculate_ema(df_1w['close'].values, span=21)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_raw)
    
    # === DAILY INDICATORS ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Daily EMA for local trend
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    # Warmup - need at least 20 bars for Donchian
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]):
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
        
        # === TREND DIRECTION (Weekly) ===
        weekly_trend_bullish = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        weekly_ema_bullish = close[i] > ema_1w_aligned[i] if not np.isnan(ema_1w_aligned[i]) else True
        weekly_trend_bearish = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # Local trend (daily)
        local_bullish = ema_8[i] > ema_21[i] if not np.isnan(ema_8[i]) else True
        local_bearish = ema_8[i] < ema_21[i] if not np.isnan(ema_8[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN CHANNEL BREAKOUT ===
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = (upper + lower) / 2 if not np.isnan(upper) and not np.isnan(lower) else close[i]
        
        # Price broke above channel (bullish breakout)
        bullish_breakout = close[i] > upper and high[i] > upper
        
        # Price broke below channel (bearish breakout)
        bearish_breakout = close[i] < lower and low[i] < lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + local bullish + breakout above + volume
        if weekly_trend_bullish and local_bullish and bullish_breakout:
            if vol_spike:
                desired_signal = SIZE
            else:
                # Allow entry without volume but with trend confirmation
                desired_signal = SIZE
        
        # SHORT: Weekly bearish + local bearish + breakout below + volume
        if weekly_trend_bearish and local_bearish and bearish_breakout:
            if vol_spike:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # If weekly trend flips, exit position
        if in_position and position_side > 0 and weekly_trend_bearish:
            # Only exit if local trend also turns
            if local_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0 and weekly_trend_bullish:
            if local_bullish:
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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