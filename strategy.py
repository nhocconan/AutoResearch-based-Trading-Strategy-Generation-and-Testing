#!/usr/bin/env python3
"""
Experiment #022: 12h Camarilla ATR Breakout + Choppiness Regime + Volume

HYPOTHESIS: Apply proven Camarilla pivot pattern to 12h timeframe with
simplified entry logic. The DB shows 4h Camarilla achieved test Sharpe 1.47
on ETHUSDT. 12h should have fewer trades but better signal quality.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above H3 Camarilla level + volume spike + daily bull → long
- Bear: Price breaks below L3 Camarilla level + volume spike + daily bear → short
- Range/Chop: Choppiness > 61.8 = no entry (avoid whipsaws)
- ATR-based stops adapt to volatility in both directions

KEY INSIGHT: "ONE strong signal + volume + regime filter = fewer trades = less fee drag"
Simplicity beats complexity. The best DB strategies use minimal conditions.

TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_chop_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CI)
    CI > 61.8 = choppy/range market (mean reversion)
    CI < 38.2 = trending market (trend following)
    """
    n = len(close)
    ci = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            ci[i] = 100 * (np.log10(atr_sum) / np.log10(range_sum))
    
    return ci

def calculate_camarilla_levels(high, low, close, atr, period=14):
    """
    Camarilla Pivot Levels based on previous period's range
    Classic levels: H1-H4, L1-L4, pivot
    """
    n = len(close)
    h4 = np.full(n, np.nan)
    h3 = np.full(n, np.nan)
    h2 = np.full(n, np.nan)
    h1 = np.full(n, np.nan)
    l1 = np.full(n, np.nan)
    l2 = np.full(n, np.nan)
    l3 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    
    for i in range(1, n):
        range_val = high[i-1] - low[i-1]
        if range_val > 0:
            pivot[i] = (high[i-1] + low[i-1] + close[i-1]) / 3
            
            # Camarilla equations (classic)
            h4[i] = close[i-1] + range_val * 1.1 / 2
            h3[i] = close[i-1] + range_val * 1.1 / 4
            h2[i] = close[i-1] + range_val * 1.1 / 6
            h1[i] = close[i-1] + range_val * 1.1 / 12
            
            l1[i] = close[i-1] - range_val * 1.1 / 12
            l2[i] = close[i-1] - range_val * 1.1 / 6
            l3[i] = close[i-1] - range_val * 1.1 / 4
            l4[i] = close[i-1] - range_val * 1.1 / 2
    
    return h4, h3, h2, h1, l1, l2, l3, l4, pivot

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, mid, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === Daily (HTF) trend: EMA comparison for direction ===
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Daily trend: bull if EMA21 > EMA50, bear if EMA21 < EMA50
    htf_bull = ema_21_1d > ema_50_1d
    htf_bear = ema_21_1d < ema_50_1d
    
    # Align HTF to 12h
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bull.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bear.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Choppiness Index
    ci = calculate_choppiness(high, low, close, period=14)
    
    # Camarilla levels
    h4, h3, h2, h1, l1, l2, l3, l4, pivot = calculate_camarilla_levels(high, low, close, atr_14)
    
    # Donchian for breakout confirmation
    dc_upper_20, _, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 50  # Enough for indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ci[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(h3[i]) or np.isnan(l3[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # CI > 61.8 = choppy (skip entries), CI < 50 = trending (allow entries)
        is_choppy = ci[i] > 61.8
        is_trending = ci[i] < 50.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === HTF TREND ===
        htf_bull_trend = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear_trend = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # === CAMARILLA SIGNALS ===
        # Long: price touches/breaks H3 with momentum, in uptrend
        # Short: price touches/breaks L3 with momentum, in downtrend
        
        bull_cam_signal = (close[i] >= h3[i] - atr_14[i] * 0.1)  # Near or above H3
        bear_cam_signal = (close[i] <= l3[i] + atr_14[i] * 0.1)  # Near or below L3
        
        # Donchian breakout confirmation
        bull_dc_confirm = close[i] >= dc_upper_20[i] - atr_14[i] * 0.2 if not np.isnan(dc_upper_20[i]) else False
        bear_dc_confirm = close[i] <= dc_lower_20[i] + atr_14[i] * 0.2 if not np.isnan(dc_lower_20[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG ENTRY: Camarilla H3 touch + volume + daily bull + not choppy
            if bull_cam_signal and vol_spike and htf_bull_trend and not is_choppy:
                desired_signal = SIZE
            
            # SHORT ENTRY: Camarilla L3 touch + volume + daily bear + not choppy
            elif bear_cam_signal and vol_spike and htf_bear_trend and not is_choppy:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if regime turns choppy
                if is_choppy and ci[i] > 65:
                    desired_signal = 0.0
                
                # Exit if daily trend turns bear
                if htf_bear_trend:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if regime turns choppy
                if is_choppy and ci[i] > 65:
                    desired_signal = 0.0
                
                # Exit if daily trend turns bull
                if htf_bull_trend:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals