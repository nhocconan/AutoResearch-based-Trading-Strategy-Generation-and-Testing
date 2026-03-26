#!/usr/bin/env python3
"""
Experiment #027: 1d Donchian Breakout + Volume Spike

HYPOTHESIS: 1d timeframe naturally limits trade frequency. Using 20-period
Donchian breakouts (proven in DB with SOL Sharpe 1.10-1.38) with ONLY volume
confirmation as filter. Simple = robust. No additional trend/regime filters
to avoid overtrading.

Entry: Price closes above/below 20d Donchian with 1.5x volume spike.
Exit: 2.5 ATR stoploss + 3R take-profit trailing stop.
Size: 0.25 (discrete, conservative).

TARGET: 75-150 total trades over 4 years (19-37/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_simple_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
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
    
    # === Calculate 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative for 1d
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    profit_target = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Long: price breaks above 20d high + volume spike
            if price_above_upper and vol_spike:
                desired_signal = SIZE
            
            # Short: price breaks below 20d low + volume spike
            if price_below_lower and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS AND TAKE-PROFIT ===
        stoploss_triggered = False
        takeprofit_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
            # Take profit: trail when 2R profit reached
            if close[i] >= profit_target:
                # Trail stop at 1.5R
                new_stop = entry_price + 1.5 * (highest_since_entry - entry_price)
                stop_price = max(stop_price, new_stop)
            if low[i] < stop_price:
                takeprofit_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
            # Take profit: trail when 2R profit reached
            if close[i] <= profit_target:
                # Trail stop at 1.5R
                new_stop = entry_price - 1.5 * (entry_price - lowest_since_entry)
                stop_price = min(stop_price, new_stop)
            if high[i] > stop_price:
                takeprofit_triggered = True
        
        if stoploss_triggered or takeprofit_triggered:
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
                profit_target = entry_price + 3.0 * entry_atr * position_side
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
                profit_target = 0.0
        
        signals[i] = desired_signal
    
    return signals