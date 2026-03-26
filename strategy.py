#!/usr/bin/env python3
"""
Experiment #013: 4h Donchian Breakout + 12h HMA Trend + Volume + ADX

HYPOTHESIS: Donchian(20) breakouts capture momentum moves, but only work when:
1. 12h HMA confirms trend direction (multi-timeframe alignment)
2. Volume spike confirms institutional participation
3. ADX > 25 confirms trending regime (not chop)

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Long breakouts above Donchian high when 12h HMA bullish
- Bear: Short breakouts below Donchian low when 12h HMA bearish
- ADX filters out choppy periods where breakouts fail (whipsaws)
- Volume confirms real moves vs fake breakouts

TARGET: 75-200 total trades over 4 years (~20-50/year)
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (SOL Sharpe=1.382, 95tr)

KEY DESIGN:
1. Donchian(20) breakout = primary signal
2. 12h HMA(21) = trend bias (only trade in HTF direction)
3. Volume > 1.5x 20-bar MA = confirmation
4. ADX(14) > 25 = trending regime filter
5. ATR(14) stoploss = 2.5x ATR trailing
6. Signal: 0.30 discrete, hold until stop/TP

POSITION MANAGEMENT:
- Enter on breakout + all filters
- Hold position until stoploss OR take profit OR opposite signal
- Do NOT exit just because entry signal disappears (massive churn killer)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma12_vol_adx_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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
    adx_14 = calculate_adx(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking - CRITICAL: hold until stop/TP, not every bar
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            continue
        
        # === REGIME CHECK ===
        is_trending = adx_14[i] > 25.0  # Only trade in trending markets
        
        # === TREND BIAS (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout = close crosses above upper or below lower
        prev_upper = donch_upper[i-1] if i > 0 else donch_upper[i]
        prev_lower = donch_lower[i-1] if i > 0 else donch_lower[i]
        
        breakout_long = close[i] > prev_upper and close[i-1] <= prev_upper
        breakout_short = close[i] < prev_lower and close[i-1] >= prev_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Donchian breakout + bullish 12h HMA + volume
            if breakout_long and price_above_12h_hma and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Donchian breakout + bearish 12h HMA + volume
            if breakout_short and not price_above_12h_hma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (trailing) ===
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
        
        # === TAKE PROFIT (opposite Donchian band) ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at lower Donchian (mean reversion after run)
            if donch_lower[i] > 0 and high[i] >= donch_lower[i] * 1.02:
                # Check if we've run 2R
                profit = (high[i] - entry_price) / entry_atr
                if profit >= 2.5:
                    tp_triggered = True
        
        if in_position and position_side < 0:
            if donch_upper[i] > 0 and low[i] <= donch_upper[i] * 0.98:
                profit = (entry_price - low[i]) / entry_atr
                if profit >= 2.5:
                    tp_triggered = True
        
        # === OPPOSITE SIGNAL EXIT ===
        opposite_signal = False
        if in_position and position_side > 0 and desired_signal < 0:
            opposite_signal = True
        if in_position and position_side < 0 and desired_signal > 0:
            opposite_signal = True
        
        # === APPLY EXITS ===
        if stoploss_triggered or tp_triggered or opposite_signal:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            stop_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        elif desired_signal != 0.0 and not in_position:
            # New entry
            signals[i] = desired_signal
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
        elif in_position:
            # Hold position - keep same signal
            signals[i] = float(position_side) * SIZE
    
    return signals