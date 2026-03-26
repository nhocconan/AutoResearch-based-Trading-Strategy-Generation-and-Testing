#!/usr/bin/env python3
"""
Experiment #024: 6h Ichimoku Cloud + 1d Trend Filter

HYPOTHESIS: Ichimoku Cloud system captures institutional price structure.
TK cross (Tenkan/Kijun) signals momentum shifts. Cloud (Kumo) provides
dynamic support/resistance. Combined with 1d HMA for trend direction,
this works in BOTH bull markets (long TK crosses above cloud) and
bear markets (short TK crosses below cloud).

TIMEFRAME: 6h primary
HTF: 1d for trend bias
TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52):
    """
    Ichimoku Cloud calculation
    Returns: tenkan, kijun, senkou_a, senkou_b, close (for chikou comparison)
    """
    n = len(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = np.full(n, np.nan, dtype=np.float64)
    for i in range(8, n):
        period_high = np.max(high[i-8:i+1])
        period_low = np.min(low[i-8:i+1])
        tenkan[i] = (period_high + period_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = np.full(n, np.nan, dtype=np.float64)
    for i in range(25, n):
        period_high = np.max(high[i-25:i+1])
        period_low = np.min(low[i-25:i+1])
        kijun[i] = (period_high + period_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = np.full(n, np.nan, dtype=np.float64)
    for i in range(25, n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    senkou_b_arr = np.full(n, np.nan, dtype=np.float64)
    for i in range(51, n):
        period_high = np.max(high[i-51:i+1])
        period_low = np.min(low[i-51:i+1])
        senkou_b_arr[i] = (period_high + period_low) / 2.0
    
    return tenkan, kijun, senkou_a, senkou_b_arr

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h Ichimoku
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR for stops
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
        
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr_smooth + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr_smooth + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx = calculate_adx(high, low, close)
    
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
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
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
        
        # === ICHIMOKU SIGNALS ===
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] if i > 1 else False
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] if i > 1 else False
        
        # Cloud boundaries (average of senkou a and b)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # Price position relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        price_in_cloud = not price_above_cloud and not price_below_cloud
        
        # TK line position
        tk_above_kijun = tenkan[i] > kijun[i]
        
        # === 1d TREND ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === ADX TREND STRENGTH ===
        adx_val = adx[i] if not np.isnan(adx[i]) else 0
        adx_strong = adx_val > 18
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === TK CROSS UP (BULLISH) ===
            # Only in bull trend (price above 1d HMA) AND either above cloud or ADX strong
            if tk_cross_up and price_above_1d_hma:
                if (price_above_cloud or (adx_strong and not price_below_cloud)):
                    if vol_spike or adx_strong:
                        desired_signal = SIZE
            
            # === TK CROSS DOWN (BEARISH) ===
            # Only in bear trend (price below 1d HMA) AND either below cloud or ADX strong
            if tk_cross_down and not price_above_1d_hma:
                if (price_below_cloud or (adx_strong and not price_above_cloud)):
                    if vol_spike or adx_strong:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === TK REVERSAL EXIT ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long on TK cross down
            if tk_cross_down:
                exit_triggered = True
            # Or price falls into cloud in weak ADX
            if price_below_cloud and not adx_strong:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short on TK cross up
            if tk_cross_up:
                exit_triggered = True
            # Or price rises into cloud in weak ADX
            if price_above_cloud and not adx_strong:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
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