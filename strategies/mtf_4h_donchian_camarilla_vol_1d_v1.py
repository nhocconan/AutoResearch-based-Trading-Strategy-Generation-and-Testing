#!/usr/bin/env python3
"""
Experiment #023: 4h Donchian Breakout + Camarilla Zone + Volume + 1d EMA

HYPOTHESIS: Donchian(20) captures trend momentum breakouts while Camarilla
levels add a mean-reversion overlay — price reaching a Camarilla S/R after
a Donchian breakout = high probability continuation. This dual-structure
approach catches larger moves while avoiding whipsaws.

WHY 4h (not 12h): DB top performers use 4h. 12h was too slow (#005).
Donchian(20) on 4h = 5-day breakout window — captures multi-day swings.

WHY IT WORKS: Donchian breakout = trend acceleration. Camarilla S3/S4 = 
institutional zones. Volume confirms institutional participation. 1d EMA
filters counter-trend entries.

TARGET: 100-200 total trades over 4 years (25-50/year).
Signal size: 0.25-0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_camarilla_vol_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values > 61.8 = choppy/range, < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20 periods = 5 days)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    SIZE_HALF = 0.125
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    profit_taken = False
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === CHOPPINESS REGIME (use as soft filter, not hard block) ===
        chop = chop_14[i]
        in_chop = chop > 61.8 if not np.isnan(chop) else False
        in_trend = chop < 38.2 if not np.isnan(chop) else False
        
        # === CAMARILLA LEVELS from previous bar ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # Camarilla middle zone (between S3 and R3)
        in_camarilla_zone = (close[i] >= s3 and close[i] <= r3)
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout = close above highest high of last 20 bars (for longs)
        #            OR close below lowest low of last 20 bars (for shorts)
        donchian_broken_up = close[i] > highest_high[i - 1]  # shift(1) for no look-ahead
        donchian_broken_down = close[i] < lowest_low[i - 1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Condition: Uptrend (price > 1d EMA) + Donchian breakout + Volume
            # OR: Price at/near Camarilla S3-S4 zone in uptrend with volume
            if price_above_1d_ema:
                # Primary: Donchian breakout
                if donchian_broken_up and vol_spike:
                    desired_signal = SIZE
                # Secondary: Camarilla S3/S4 bounce in uptrend
                elif vol_spike and (low[i] <= s4 or (low[i] <= s3 and not in_chop)):
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            if not price_above_1d_ema:
                # Primary: Donchian breakdown
                if donchian_broken_down and vol_spike:
                    desired_signal = -SIZE
                # Secondary: Camarilla R3/R4 rejection in downtrend
                elif vol_spike and (high[i] >= r4 or (high[i] >= r3 and not in_chop)):
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT at 2R + half position ===
        bars_held = i - entry_bar
        if in_position and not profit_taken and bars_held >= 2:
            if position_side > 0:
                profit_2r = entry_price + 2.0 * entry_atr
                if high[i] >= profit_2r:
                    desired_signal = SIZE_HALF  # Take half profit
                    profit_taken = True
            elif position_side < 0:
                profit_2r = entry_price - 2.0 * entry_atr
                if low[i] <= profit_2r:
                    desired_signal = -SIZE_HALF
                    profit_taken = True
        
        # === HOLD MINIMUM 2 bars to avoid fee churn ===
        if in_position and bars_held < 2:
            # Keep current signal to avoid early exit
            if position_side > 0:
                desired_signal = SIZE
            elif position_side < 0:
                desired_signal = -SIZE
        
        # === UPDATE POSITION ===
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
                profit_taken = False
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals