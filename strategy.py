#!/usr/bin/env python3
"""
Experiment #003 Retry: 4h Donchian Breakout + 12h Trend + Volume Filter

HYPOTHESIS: Donchian channel breakouts capture momentum shifts. 12h HMA provides 
trend bias to avoid counter-trend breakouts. Volume confirmation filters false 
breakouts. Simple 2-3 condition entry = adequate trade frequency (75-150 over 4 years).

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakout works in both directions (long on upper break, short on lower)
- 12h HMA trend filter prevents counter-trend entries (long only when price>12h_HMA)
- Volume spike confirms institutional participation, not retail FOMO
- ATR stoploss protects against false breakouts (common in crypto)

TARGET: 75-150 total trades over 4 years (proven pattern from DB winners).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95 trades)

KEY DESIGN:
1. Donchian(20) breakout as primary signal
2. 12h HMA(21) for trend bias (only long above, short below)
3. Volume > 1.2x 20-MA (moderate confirmation, not extreme)
4. ATR(14) stoploss at 2.5x
5. Signal: 0.30 (discrete)

LESSONS FROM FAILURES:
- #007: 624 trades = way too many (overtrading)
- #009/#013/#016: Camarilla too restrictive (22-58 trades = too few)
- #010/#014/#017/#018/#020: 0 trades = conditions mutually exclusive
- Current code: Too many confluence filters = too few trades

SIMPLIFICATION: Remove choppiness filter, remove EMA cross, remove pivot distance logic.
Just: Donchian break + 12h trend + volume confirm.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_trend_vol_v2"
timeframe = "4h"
leverage = 1.0

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h data for trend bias
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA
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
    
    # Warmup for Donchian(20) + ATR(14)
    warmup = 40
    
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
        
        if np.isnan(hma_12h_aligned[i]):
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
        
        # === TREND BIAS (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        
        # === VOLUME CONFIRMATION (moderate, not extreme) ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Breakout above Donchian + price above 12h HMA + volume confirm
        if breakout_long and price_above_12h_hma and vol_confirm:
            desired_signal = SIZE
        
        # SHORT: Breakout below Donchian + price below 12h HMA + volume confirm
        if breakout_short and not price_above_12h_hma and vol_confirm:
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
        
        # === TAKE PROFIT at opposite Donchian band ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP when price reaches lower band (trend reversal)
            if close[i] < donchian_lower[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP when price reaches upper band (trend reversal)
            if close[i] > donchian_upper[i]:
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