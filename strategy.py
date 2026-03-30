#!/usr/bin/env python3
"""
Experiment #028: 6h Ichimoku Cloud + Choppiness Regime + 1d Trend

HYPOTHESIS: Ichimoku is a COMPLETE trading system (trend + momentum + structure)
that was designed for Japanese equity volatility - similar to crypto. Unlike Donchian
which just tracks extremes, Ichimoku's dual signal lines (Tenkan/Kijun) + cloud
structure provides more nuanced entries with built-in false signal filtering.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Tenkan crosses above Kijun + price above cloud = momentum confirmation
- Bear: Tenkan crosses below Kijun + price below cloud = breakdown confirmation
- Choppiness keeps us out of range-bound zones where Ichimoku produces whipsaws
- 1d SMA50 confirms major trend direction

WHY 6h: Faster than 12h (more signals), slower than 4h (less fee drag).
Ichimoku parameters (9,26,52) on 6h captures ~2-4 day trends.

TARGET: 75-150 total trades over 4 years. HARD MAX: 300.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_chop_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52):
    """
    Calculate Ichimoku Cloud components.
    Returns: tenkan_sen, kijun_sen, senkou_a, senkou_b, cloud_upper, cloud_lower
    """
    n = len(close)
    
    # Initialize arrays
    tenkan_sen = np.full(n, np.nan)
    kijun_sen = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    cloud_upper = np.full(n, np.nan)
    cloud_lower = np.full(n, np.nan)
    
    # Calculate Tenkan-sen (fastest)
    for i in range(tenkan - 1, n):
        period_high = np.max(high[i - tenkan + 1:i + 1])
        period_low = np.min(low[i - tenkan + 1:i + 1])
        tenkan_sen[i] = (period_high + period_low) / 2.0
    
    # Calculate Kijun-sen (slower)
    for i in range(kijun - 1, n):
        period_high = np.max(high[i - kijun + 1:i + 1])
        period_low = np.min(low[i - kijun + 1:i + 1])
        kijun_sen[i] = (period_high + period_low) / 2.0
    
    # Calculate Senkou Span A (leading span A) - plotted 26 periods ahead
    for i in range(kijun - 1, n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2.0
    
    # Calculate Senkou Span B (leading span B) - plotted 26 periods ahead
    for i in range(senkou_b - 1, n):
        period_high = np.max(high[i - senkou_b + 1:i + 1])
        period_low = np.min(low[i - senkou_b + 1:i + 1])
        senkou_b[i] = (period_high + period_low) / 2.0
    
    # Cloud is midpoint of Senkou A and Senkou B
    for i in range(kijun - 1, n):
        if not np.isnan(senkou_a[i]) and not np.isnan(senkou_b[i]):
            cloud_upper[i] = max(senkou_a[i], senkou_b[i])
            cloud_lower[i] = min(senkou_a[i], senkou_b[i])
    
    return tenkan_sen, kijun_sen, senkou_a, senkou_b, cloud_upper, cloud_lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

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
    
    # 1d SMA50 for major trend direction
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Ichimoku Cloud (9, 26, 52 periods)
    tenkan, kijun, senkou_a, senkou_b, cloud_upper, cloud_lower = calculate_ichimoku(
        high, low, close, tenkan=9, kijun=26, senkou_b=52
    )
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    prev_tenkan = np.nan
    prev_kijun = np.nan
    
    warmup = 60  # Need 52 for Ichimoku Senkou B + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            prev_tenkan = np.nan
            prev_kijun = np.nan
            continue
        
        if np.isnan(sma_50_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            prev_tenkan = np.nan
            prev_kijun = np.nan
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            prev_tenkan = np.nan
            prev_kijun = np.nan
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_50_1d_aligned[i]
        
        # === REGIME CHECK (Choppiness) ===
        # Skip if too choppy (CHOP > 61.8)
        if chop[i] > 61.8 and not in_position:
            signals[i] = 0.0
            prev_tenkan = tenkan[i]
            prev_kijun = kijun[i]
            continue
        
        # === CLOUD POSITION CHECK ===
        # Price must be on correct side of cloud for entry
        cloud_ready = not np.isnan(cloud_upper[i]) and not np.isnan(cloud_lower[i])
        if not cloud_ready:
            signals[i] = 0.0
            prev_tenkan = tenkan[i]
            prev_kijun = kijun[i]
            continue
        
        price_above_cloud = close[i] > cloud_upper[i]
        price_below_cloud = close[i] < cloud_lower[i]
        
        # === MOMENTUM CHECK ===
        # Check for Tenkan/Kijun cross (current and previous)
        current_tenkan = tenkan[i]
        current_kijun = kijun[i]
        
        bullish_cross = (prev_tenkan <= prev_kijun) and (current_tenkan > current_kijun)
        bearish_cross = (prev_tenkan >= prev_kijun) and (current_tenkan < current_kijun)
        
        # Volume confirmation (not required but helps)
        vol_spike = vol_ratio[i] > 1.5
        
        # === ICHIMOKU SIGNALS ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Tenkan crosses above Kijun + price above cloud + uptrend confirmed
            if bullish_cross and price_above_cloud and price_above_1d_sma:
                if vol_spike or chop[i] < 50:  # Volume spike or trending
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Tenkan crosses below Kijun + price below cloud + downtrend confirmed
            if bearish_cross and price_below_cloud and not price_above_1d_sma:
                if vol_spike or chop[i] < 50:
                    desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.0 ATR) ===
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
        
        # === SOFT EXIT: Reverse signal ===
        if in_position and position_side > 0:
            if bearish_cross and close[i] < cloud_lower[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if bullish_cross and close[i] > cloud_upper[i]:
                desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days on 6h) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if momentum weakens
            if position_side > 0 and current_tenkan < current_kijun:
                desired_signal = 0.0
            if position_side < 0 and current_tenkan > current_kijun:
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
                entry_bar = i
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
        
        # Update previous Tenkan/Kijun for cross detection
        prev_tenkan = current_tenkan
        prev_kijun = current_kijun
        
        signals[i] = desired_signal
    
    return signals