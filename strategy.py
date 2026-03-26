Looking at the failures:
- 4h strategies are uniformly failing (overtrading or negative Sharpe)
- 12h has 54% keep rate in experiments
- mtf_12h_donchian_vol_chop_simple_v1 was best but overtraded (323 trades)
- Need to tighten entry conditions to get 50-150 trades

Key insight: Use 1w data for trend direction (most reliable), 12h for entries. WMA(21) on 1w provides cleaner trend signal than HMA. Use Williams %R extremes (<-80 or >-20) but only when confirmed by 1w trend.

#!/usr/bin/env python3
"""
Experiment #008: 12h Williams %R + 1w Donchian + Volume Spike

HYPOTHESIS: Williams %R captures momentum extremes. Using 1w Donchian 
for trend filters out counter-trend trades. Volume spike confirms institutional 
involvement. 12h timeframe ensures enough trades without overtrading.

TARGET: 75-150 total trades over 4 years (18-37/year on 12h).
Reference: mtf_12h_donchian_vol_chop_simple_v1 (best in session but overtraded)

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Long when Williams %R <-80 + price above 1w Donchian mid
- Bear: Short when Williams %R >-20 + price below 1w Donchian mid
- 1w Donchian adapts to regime, reducing bear market losses

KEY DESIGN:
1. 1w Donchian channel for trend (aligns with 12h, shifts 1 bar)
2. Williams %R(14) extremes for entry timing
3. Volume spike >1.5x 20-avg for confirmation
4. ATR-based stoploss (2x ATR)
5. Discrete signal: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_donchian_vol_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 1e-10:
            willr[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, middle, lower

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

def calculate_wma(close, period):
    """Weighted Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = np.sum(weights)
    
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1].astype(np.float64)
        if not np.any(np.isnan(window)):
            result[i] = np.sum(window * weights) / weight_sum
    
    return result

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Donchian (shifted by 1 to avoid look-ahead)
    dc_upper_1w, dc_mid_1w, dc_lower_1w = calculate_donchian(
        df_1w['high'].values, df_1w['low'].values, period=20
    )
    dc_upper_aligned = align_htf_to_ltf(prices, df_1w, dc_upper_1w)
    dc_mid_aligned = align_htf_to_ltf(prices, df_1w, dc_mid_1w)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1w, dc_lower_1w)
    
    # 1w WMA for additional trend confirmation
    wma_1w = calculate_wma(df_1w['close'].values, period=21)
    wma_aligned = align_htf_to_ltf(prices, df_1w, wma_1w)
    
    # Calculate 12h indicators
    willr_14 = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    # Warmup - need 1w data (12h bars for ~2 weeks)
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(willr_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(dc_mid_aligned[i]) or np.isnan(wma_aligned[i]):
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
        
        # === TREND CHECK (1w) ===
        price_above_1w_mid = close[i] > dc_mid_aligned[i]
        price_above_1w_wma = close[i] > wma_aligned[i]
        bullish_trend = price_above_1w_mid and price_above_1w_wma
        
        price_below_1w_mid = close[i] < dc_mid_aligned[i]
        price_below_1w_wma = close[i] < wma_aligned[i]
        bearish_trend = price_below_1w_mid and price_below_1w_wma
        
        # === WILLIAMS %R EXTREMES ===
        willr = willr_14[i]
        oversold = willr < -80  # Strong momentum shift up
        overbought = willr > -20  # Strong momentum shift down
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Oversold + bullish trend + volume
        if oversold and bullish_trend:
            if vol_spike:
                desired_signal = SIZE
            else:
                # Still enter without volume if trend is strong
                desired_signal = SIZE
        
        # SHORT: Overbought + bearish trend + volume
        if overbought and bearish_trend:
            if vol_spike:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE
        
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
            desired_signal = 0.0
        
        # === TAKE PROFIT at 1w Donchian opposite band ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at 1w upper band
            if not np.isnan(dc_upper_aligned[i]) and high[i] >= dc_upper_aligned[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at 1w lower band
            if not np.isnan(dc_lower_aligned[i]) and low[i] <= dc_lower_aligned[i]:
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