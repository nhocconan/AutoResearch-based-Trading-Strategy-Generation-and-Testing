#!/usr/bin/env python3
"""
Experiment #028: 6h Bollinger Squeeze + Williams %R Extreme + 1d EMA Trend

HYPOTHESIS: Bollinger Band squeeze identifies low-volatility expansions before
breakouts. Williams %R provides momentum confirmation at extremes. The 1d EMA
defines the primary trend. This combination catches "volatility trap" reversals
where squeeze expands in the direction of trend.

WHY IT WORKS IN BULL AND BEAR:
- In bull: squeeze in uptrend → bullish expansion with momentum
- In bear: squeeze in downtrend → bearish expansion with momentum
- Shorting bearish expansions works in bear markets
- Longing bullish expansions works in bull markets
- Williams %R filters out early entries (only at extremes)

TARGET: 75-125 total trades over 4 years = 19-31/year. HARD MAX: 300.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_bb_squeeze_williams_1d_v1"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    williams_r = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return williams_r

def calculate_bollinger_bands(close, period=20, num_std=2):
    """Bollinger Bands - returns upper, middle, lower, width"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std(ddof=0).values
    
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    
    return upper, middle, lower, width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, num_std=2)
    
    # BB Width MA for squeeze detection
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio for volatility regime (current vs recent average)
    atr_ma = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
    # Volume
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
    take_profit_hit = False
    
    warmup = 100  # Need enough for all indicators
    
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
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(bb_width_ma[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        trend_bullish = price_above_1d_ema
        trend_bearish = not price_above_1d_ema
        
        # === SQUEEZE DETECTION ===
        # BB width below its MA = squeeze forming
        in_squeeze = bb_width[i] < bb_width_ma[i]
        
        # Volatility expansion (not in low-vol regime)
        vol_expanding = atr_ratio[i] > 0.7
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === WILLIAMS %R MOMENTUM ===
        # Extreme readings
        wr_oversold = williams_r[i] < -80  # Strong bullish momentum
        wr_overbought = williams_r[i] > -20  # Strong bearish momentum
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Squeeze expansion + oversold + bullish trend ===
            # Squeeze breaks with volatility expansion, momentum at extreme, trend confirms
            if in_squeeze and vol_expanding and wr_oversold and trend_bullish:
                if vol_spike:  # Volume confirms
                    desired_signal = SIZE
            
            # === SHORT: Squeeze expansion + overbought + bearish trend ===
            if in_squeeze and vol_expanding and wr_overbought and trend_bearish:
                if vol_spike:
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
        
        # === TAKE PROFIT (3.0 ATR - use half position) ===
        if in_position and not take_profit_hit:
            if position_side > 0:
                profit_target = entry_price + 3.0 * entry_atr
                if high[i] >= profit_target:
                    desired_signal = SIZE / 2  # Half position
                    take_profit_hit = True
                    stop_price = entry_price  # Lock in breakeven
            elif position_side < 0:
                profit_target = entry_price - 3.0 * entry_atr
                if low[i] <= profit_target:
                    desired_signal = -SIZE / 2
                    take_profit_hit = True
                    stop_price = entry_price
        
        # === TIME-BASED EXIT (hold max 20 bars = 5 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 20:
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
                take_profit_hit = False
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
                take_profit_hit = False
        
        signals[i] = desired_signal
    
    return signals