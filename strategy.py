#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian Breakout + Volume + 1d EMA (Simplified)

HYPOTHESIS: Remove the Camarilla secondary entry that caused #003 to overtrade
(915 trades). Keep only: Donchian(20) breakout + volume spike + 1d EMA trend.
This should produce 100-200 total trades (vs 915 in #003) with better Sharpe.

WHY IT SHOULD WORK IN BULL AND BEAR:
- Bull: Donchian breakout upward catches momentum continuation
- Bear: Donchian breakdown with trend filter catches short opportunities
- 1d EMA ensures we only trade with the major trend
- Volume confirms institutional participation
- ATR trailing stop manages risk in both directions

TARGET: 100-200 total trades over 4 years (25-50/year)
Signal size: 0.25
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_volume_ema_simplified_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = 5 days on 4h)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness Index (soft filter only)
    chop = np.full(n, np.nan)
    chop_period = 14
    for i in range(chop_period, n):
        atr_sum = 0.0
        for j in range(i - chop_period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        hh = max(high[i - chop_period + 1:i + 1])
        ll = min(low[i - chop_period + 1:i + 1])
        range_sum = hh - ll
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(chop_period))
    
    # Signals
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
    profit_taken = False
    
    warmup = max(100, donchian_period + 20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation (1.5x average)
        vol_spike = vol_ratio[i] > 1.5
        
        # Choppiness soft filter (not hard block)
        in_chop = chop[i] > 61.8 if not np.isnan(chop[i]) else False
        
        # === DONCHIAN BREAKOUT (shift by 1 to avoid look-ahead) ===
        donchian_broken_up = close[i] > highest_high[i - 1]
        donchian_broken_down = close[i] < lowest_low[i - 1]
        
        # === ENTRY LOGIC (SIMPLIFIED - only Donchian breakout) ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Donchian breakout UP + volume spike + in uptrend
            if price_above_1d_ema and donchian_broken_up and vol_spike:
                desired_signal = SIZE
            # Alternative: in strong uptrend (no chop) + breakout even without vol spike
            elif price_above_1d_ema and donchian_broken_up and not in_chop:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Donchian breakdown DOWN + volume spike + in downtrend
            if price_below_1d_ema and donchian_broken_down and vol_spike:
                desired_signal = -SIZE
            elif price_below_1d_ema and donchian_broken_down and not in_chop:
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
        if in_position and not profit_taken and bars_held >= 3:
            if position_side > 0:
                profit_2r = entry_price + 2.0 * entry_atr
                if high[i] >= profit_2r:
                    desired_signal = SIZE / 2  # Take half profit
                    profit_taken = True
            elif position_side < 0:
                profit_2r = entry_price - 2.0 * entry_atr
                if low[i] <= profit_2r:
                    desired_signal = -SIZE / 2
                    profit_taken = True
        
        # === HOLD MINIMUM 4 bars to reduce fee churn ===
        if in_position and bars_held < 4:
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