#!/usr/bin/env python3
"""
Experiment #051: 6h Ichimoku Cloud + 1d Trend Filter + Volume Confirmation

HYPOTHESIS: Ichimoku cloud (Tenkan/Kijun cross + price vs cloud) on 6h timeframe,
filtered by 1d EMA50 trend and volume confirmation (1.3x average volume),
captures strong momentum with controlled trade frequency. The cloud acts as
dynamic support/resistance, reducing whipsaws in sideways markets. Designed for
50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize
fee drag while maintaining statistical significance. Works in bull (trend
following) and bear (cloud as resistance) regimes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B"""
    n = len(close)
    if n < senkou:
        return (np.full(n, np.nan), np.full(n, np.nan), 
                np.full(n, np.nan), np.full(n, np.nan))
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() +
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() +
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() +
                 pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    return (tenkan_sen.values, kijun_sen.values, 
            senkou_a.values, senkou_b.values)

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    close = prices["close"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA50 for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # === 6h Ichimoku Cloud ===
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # === 6h ATR for stoploss ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # === 6h Volume MA for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60  # Ensure enough data for Ichimoku (52) and HTF
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_1d_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Ichimoku Components ---
        price = close[i]
        # Cloud top/bottom (Senkou Span A/B)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # --- Trend Filters ---
        # 6h: Tenkan > Kijun = bullish momentum
        momentum_bullish = tenkan[i] > kijun[i]
        momentum_bearish = tenkan[i] < kijun[i]
        
        # 6h: Price vs Cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # 1d: EMA50 trend
        trend_bullish = price > ema_1d_50_aligned[i]
        trend_bearish = price < ema_1d_50_aligned[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.3 if vol_ma_20[i] > 1e-10 else False
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: 
            # 1. Price re-enters cloud (loss of momentum)
            # 2. Tenkan/Kijun cross reverses
            # 3. 1d trend reverses (strong filter)
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price in cloud OR bearish cross OR 1d trend turns bearish
                    if (price > cloud_bottom and price < cloud_top) or \
                       (tenkan[i] < kijun[i]) or \
                       (not trend_bullish):
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price in cloud OR bullish cross OR 1d trend turns bullish
                    if (price > cloud_bottom and price < cloud_top) or \
                       (tenkan[i] > kijun[i]) or \
                       (not trend_bearish):
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions:
        # Price above cloud + bullish Tenkan/Kijun cross + bullish 1d trend + volume
        if price_above_cloud and momentum_bullish and trend_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Price below cloud + bearish Tenkan/Kijun cross + bearish 1d trend + volume
        elif price_below_cloud and momentum_bearish and trend_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals