#!/usr/bin/env python3
"""
Experiment #028: 4h Donchian Breakout + Williams %R Extreme + 1d SMA50 Trend

HYPOTHESIS: Williams %R at extremes (<-80 or >-20) identifies exhaustion points
where institutional reversals occur. Combining with Donchian breakout adds 
structural confirmation. 1d SMA50 filters for trend direction.

WHY IT WORKS IN BULL AND BEAR:
- Bull: %R oversold + Donchian breakout + price>1d SMA50 = strong long
- Bear: %R overbought + Donchian breakdown + price<1d SMA50 = strong short
- Symmetric structure works in both directions

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25. Entry needs 3+ confluence for strictness.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_williams_r_1d_v1"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(high)
    williams_r = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 0:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return williams_r

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Donchian channels (20 periods = 3.3 days on 4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
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
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_50_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Previous bar values for breakout detection
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        prev_williams_r = williams_r[i - 1] if i > 0 else -50
        
        # Trend direction
        price_above_1d_sma = close[i] > sma_50_1d_aligned[i]
        
        # Williams %R extremes
        # %R < -80 = oversold (potential reversal up)
        # %R > -20 = overbought (potential reversal down)
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        # Volume confirmation (stricter: 1.8x)
        vol_spike = vol_ratio[i] > 1.8
        
        # Current bar extremes
        current_high = high[i]
        current_low = low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Donchian breakout + Williams oversold + price above trend ===
            # Price breaks above previous 20-bar high
            # Williams %R was oversold (exhaustion)
            # Price in uptrend
            breakout_long = current_high > prev_donchian_high
            was_oversold = prev_williams_r < -70  # Check if it WAS oversold before bounce
            if breakout_long and was_oversold and price_above_1d_sma:
                if vol_spike:  # Volume confirmation required
                    desired_signal = SIZE
            
            # === SHORT: Donchian breakdown + Williams overbought + price below trend ===
            # Price breaks below previous 20-bar low
            # Williams %R was overbought (exhaustion)
            # Price in downtrend
            breakout_short = current_low < prev_donchian_low
            was_overbought = prev_williams_r > -30  # Check if it WAS overbought before drop
            if breakout_short and was_overbought and not price_above_1d_sma:
                if vol_spike:  # Volume confirmation required
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === HOLDING PERIOD EXIT (min 8 bars = 1.3 days on 4h) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if price reverts to middle of channel
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
                desired_signal = 0.0
        
        # === WILLIAMS %R REVERSAL EXIT ===
        # If %R reaches opposite extreme, close position
        if in_position and position_side > 0:
            # Was long, now overbought = exit
            if williams_r[i] > -20:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Was short, now oversold = exit
            if williams_r[i] < -80:
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