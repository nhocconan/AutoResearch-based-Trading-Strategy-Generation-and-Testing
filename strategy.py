#!/usr/bin/env python3
"""
Experiment #012: 12h Donchian Breakout + Volume Spike + Choppiness

HYPOTHESIS: Simple 12h Donchian(20) breakout captures institutional momentum.
Adding volume spike confirms the breakout isn't false. Choppiness filter (<60)
reduces whipsaws in ranging markets. Simple = reliable = enough trades.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Breakout above 20-high triggers long
- Bear markets: Breakout below 20-low triggers short  
- Range markets: Choppiness >60 skips (no trades in chop)
- ATR stoploss protects in both directions

TARGET: 75-150 total trades over 4 years.
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382)

KEY DESIGN (KISS - Keep It Simple, Stupid):
1. Donchian(20) high/low as entry trigger ONLY
2. Volume spike >1.5x 20-avg for confirmation
3. Choppiness <60 (loosened from <55)
4. ATR-based stoploss
5. Size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_simple_v1"
timeframe = "12h"
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
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (skip trades), CHOP < 60 = trending (allow trades)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - high/low over period"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Load 1d data for trend context
    df_1d = get_htf_data(prices, '1d')
    sma_50_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d_raw)
    
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
    
    # Warmup for indicators
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 60.0  # Loose filter: trending or neutral
        
        # === 1d TREND CONTEXT ===
        price_above_1d_sma50 = close[i] > sma_50_1d_aligned[i] if not np.isnan(sma_50_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN LEVELS ===
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        
        # Price proximity to channel (as ATR)
        atr = atr_14[i]
        dist_to_high = (dc_high - close[i]) / atr if atr > 0 else 999
        dist_to_low = (close[i] - dc_low) / atr if atr > 0 else 999
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Breakout above 20-high + volume + trend ok
        # Entry when price within 0.3 ATR of breaking above Donchian high
        if is_trending and dist_to_high > -0.3 and dist_to_high < 1.5:
            # Bullish: price above SMA50 OR volume spike
            if price_above_1d_sma50 or vol_spike:
                desired_signal = SIZE
        
        # SHORT: Breakdown below 20-low + volume + trend ok
        # Entry when price within 0.3 ATR of breaking below Donchian low
        if is_trending and dist_to_low > -0.3 and dist_to_low < 1.5:
            # Bearish: price below SMA50 OR volume spike
            if not price_above_1d_sma50 or vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
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
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
        
        signals[i] = desired_signal
    
    return signals