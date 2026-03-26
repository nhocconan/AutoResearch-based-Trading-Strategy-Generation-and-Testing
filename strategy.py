#!/usr/bin/env python3
"""
Experiment #003: 4h Donchian Breakout + Volume Spike + 12h HMA Trend

HYPOTHESIS: Price channel breakouts (not just "near" levels) with volume 
confirmation capture institutional moves. HTF trend filter ensures we only
trade with the higher timeframe momentum. Tight entries = fewer trades = less fee drag.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakout works in ALL markets (bull breakout up, bear breakdown down)
- Bear markets: breakdowns with 12h bearish HMA = short
- Bull markets: breakouts with 12h bullish HMA = long
- Range markets: no signals when choppy (ADX < 25)

KEY DIFFERENCES FROM FAILED ATTEMPTS:
1. Breakout requires CLOSE beyond channel (not just "near")
2. Volume spike threshold 2.0x (not 1.5x) - tighter filter
3. ADX filter (not choppiness) - proven more reliable
4. Only 12h HMA for trend (not multiple EMA crosses)
5. 12h Donchian channel (20 periods) - not 4h-only

TARGET: 75-125 total trades over 4 years (proven from DB winners).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95 trades)

DESIGN:
1. 4h Donchian(20) for entry channel
2. Price must CLOSE beyond channel (tight entry)
3. Volume spike >2.0x 20-avg (stronger filter)
4. 12h HMA(48) for trend direction
5. ADX(14) > 20 for regime (trending only)
6. ATR-based stoploss (2*ATR)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_adx_12h_v2"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - vectorized for speed"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # WMA using pandas rolling
    def wma_calc(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        result = np.full(n, np.nan, dtype=np.float64)
        
        for i in range(span - 1, n):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma_calc(close, half)
    wma_full = wma_calc(close, period)
    
    # Diff
    diff = np.where(np.isnan(wma_half) | np.isnan(wma_full), np.nan, 2.0 * wma_half - wma_full)
    
    return wma_calc(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range - vectorized"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - simplified"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    # Use rolling mean for smoothing
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * np.mean(plus_dm[i-period+1:i+1]) / atr[i]
            minus_di[i] = 100 * np.mean(minus_dm[i-period+1:i+1]) / atr[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_donchian(high, low, period):
    """Donchian Channel - upper, middle, lower"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h data for HTF trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 12h Donchian for structure
    dc_up_12h, _, dc_low_12h = calculate_donchian(
        df_12h['high'].values, 
        df_12h['low'].values, 
        period=20
    )
    dc_up_12h_aligned = align_htf_to_ltf(prices, df_12h, dc_up_12h)
    dc_low_12h_aligned = align_htf_to_ltf(prices, df_12h, dc_low_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # 4h Donchian for entry
    dc_up_4h, _, dc_low_4h = calculate_donchian(high, low, period=20)
    
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
    
    # Warmup - need 20 periods for Donchian + 48 for 12h HMA
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(dc_up_4h[i]) or np.isnan(dc_low_4h[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            continue
        
        # === REGIME CHECK ===
        adx = adx_14[i]
        is_trending = adx > 20.0  # ADX > 20 = trending market
        
        # === HTF TREND DIRECTION ===
        hma_12h = hma_12h_aligned[i]
        price_above_12h_hma = close[i] > hma_12h
        price_below_12h_hma = close[i] < hma_12h
        
        # 12h channel structure
        dc_up_12h_val = dc_up_12h_aligned[i]
        dc_low_12h_val = dc_low_12h_aligned[i]
        
        # === VOLUME CONFIRMATION (tighter = 2.0x) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === DONCHIAN CHANNEL VALUES ===
        upper_4h = dc_up_4h[i]
        lower_4h = dc_low_4h[i]
        
        # === STOPLOSS CHECK ===
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
            in_position = False
            position_side = 0
            signals[i] = 0.0
            continue
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only look for entries when not in position
        if not in_position and is_trending:
            # LONG: Price CLOSES above upper Donchian + bullish HTF
            if close[i] > upper_4h and price_above_12h_hma:
                if vol_spike:
                    desired_signal = SIZE
                elif adx > 25.0:  # Strong trend doesn't need vol confirmation
                    desired_signal = SIZE
            
            # SHORT: Price CLOSES below lower Donchian + bearish HTF
            if close[i] < lower_4h and price_below_12h_hma:
                if vol_spike:
                    desired_signal = -SIZE
                elif adx > 25.0:
                    desired_signal = -SIZE
        
        # === TAKE PROFIT ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP when price retests the channel from above
            if close[i] < lower_4h:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP when price retests the channel from below
            if close[i] > upper_4h:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
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
        
        if in_position and desired_signal == 0.0:
            # Still in position, maintain signal
            desired_signal = float(position_side) * SIZE
        
        signals[i] = desired_signal
    
    return signals