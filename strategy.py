#!/usr/bin/env python3
"""
Experiment #028: 6h Donchian(50) Breakout + 12h Trend + Choppiness Regime

HYPOTHESIS: Donchian(20) on 6h generates too many signals (458-764/sym in prior attempts).
Donchian(50) = ~12-day channel = institutional breakout level. Fewer signals = less fee drag.

Key improvements:
- Donchian(50) instead of Donchian(20) - 60% fewer breakouts
- 12h HTF for BOTH trend and regime - proper MTF alignment
- Exit at Donchian mid (50% reversion) instead of holding indefinitely
- 2.0 ATR stoploss for risk management

WHY IT WORKS: Donchian channels are self-organizing support/resistance. 50-period
6h = ~12 calendar days = captures medium-term swing highs/lows. Trend filter avoids
trading against the larger move. Choppiness keeps us out of range-bound chop.

TARGET: 75-150 total trades over 4 years. HARD MAX: 300.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian50_vol_12h_trend_chop_v1"
timeframe = "6h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h SMA50 for trend direction
    sma_12h = pd.Series(df_12h['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_12h)
    
    # 12h ATR for regime (ADX-like measure)
    htf_high = df_12h['high'].values
    htf_low = df_12h['low'].values
    htf_close = df_12h['close'].values
    htf_atr = calculate_atr(htf_high, htf_low, htf_close, period=14)
    
    # 12h Choppiness on HTF
    htf_chop = calculate_choppiness(htf_high, htf_low, htf_close, period=14)
    htf_chop_aligned = align_htf_to_ltf(prices, df_12h, htf_chop)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (50 periods = ~12 days on 6h - tighter than 20 for fewer signals)
    donchian_period = 50
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation
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
    entry_bar = 0
    
    # Warmup: need Donchian(50) + buffer
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(htf_chop_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (12h SMA50) ===
        price_above_12h_sma = close[i] > sma_12h_aligned[i]
        
        # === REGIME (12h Choppiness) ===
        # Skip if too choppy (CHOP > 61.8) - only trade in trending conditions
        htf_chop_val = htf_chop_aligned[i]
        is_choppy_regime = htf_chop_val > 61.8
        
        # Skip if choppy and we're not in a position
        if is_choppy_regime and not in_position:
            signals[i] = 0.0
            continue
        
        # Previous bar's close for breakout detection (avoid look-ahead)
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Previous bar's Donchian values
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Close breaks above previous Donchian high ===
            # Price closes above previous 50-bar high with trend and volume
            if prev_close > prev_donchian_high and price_above_12h_sma and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Close breaks below previous Donchian low ===
            # Price closes below previous 50-bar low with trend and volume
            if prev_close < prev_donchian_low and not price_above_12h_sma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
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
        
        # === MID-CHANNEL EXIT (reversion to mean) ===
        if in_position and bars_held >= 4:
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
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
        
        signals[i] = desired_signal
    
    return signals