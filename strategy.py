#!/usr/bin/env python3
"""
Experiment #007: 6h Williams %R Extreme + Donchian Mean Reversion + Volatility Squeeze

HYPOTHESIS: Williams %R reaching extreme levels (-100 or 0) at Donchian channel 
boundaries marks short-term exhaustion points. By entering mean reversion trades 
ONLY when:
1. %R is at extreme (confirms oversold/overbought)
2. Price touches/fills the Donchian channel boundary (confirmed support/resistance)
3. Volume confirms the move (not just price wicking)
4. Bollinger Band width < 50th percentile (volatility squeeze = higher success rate)

WHY IT WORKS IN BULL AND BEAR: Mean reversion at extremes catches:
- Bear rallies: short at overbought %R + upper Donchian
- Bull corrections: long at oversold %R + lower Donchian
Symmetrical logic = works in both directions.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_willr_donchian_vol_squeeze_v1"
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
    """Williams %R"""
    n = len(close)
    if n < period:
        return np.full(n, -50.0)
    
    willr = np.full(n, -50.0)
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high - lowest_low > 0:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_donchian(close, period=20):
    """Donchian channel - returns upper, middle, lower"""
    n = len(close)
    upper = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(close[i - period + 1:i + 1])
        middle[i] = (upper[i] + np.min(close[i - period + 1:i + 1])) / 2
        lower[i] = np.min(close[i - period + 1:i + 1])
    
    return upper, middle, lower

def calculate_bb_width(close, period=20, num_std=2):
    """Bollinger Band width percentile rank"""
    n = len(close)
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma
    
    # Percentile rank over 100 bars
    bw_percentile = np.full(n, 50.0)
    for i in range(100, n):
        lookback = bandwidth[i - 100:i]
        valid = lookback[~np.isnan(lookback)]
        if len(valid) > 20:
            bw_percentile[i] = (valid < bandwidth[i]).sum() / len(valid) * 100
    
    return bw_percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend confirmation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_14 = calculate_williams_r(high, low, close, period=14)
    donch_upper, donch_mid, donch_lower = calculate_donchian(close, period=20)
    bw_percentile = calculate_bb_width(close, period=20, num_std=2)
    
    # Volume metrics
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 120  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_50_aligned[i]) or np.isnan(donch_lower[i]) or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_50_aligned[i]
        
        # === REGIME FILTER: Volatility squeeze (bandwidth < 50th percentile) ===
        # This ensures we're in a compressed range = higher reversal success
        in_squeeze = bw_percentile[i] < 50.0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.3
        
        # === WILLIAMS %R EXTREMES ===
        oversold = willr_14[i] < -80  # Strong oversold
        overbought = willr_14[i] > -20  # Strong overbought (remember: -20 is top)
        
        # === DONCHIAN BOUNDARIES ===
        at_lower_donch = low[i] <= donch_lower[i]  # At/slightly below lower channel
        at_upper_donch = high[i] >= donch_upper[i]  # At/slightly above upper channel
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Oversold + Lower Donchian + Volume + Squeeze (optional) ===
            # In uptrend or neutral, fade the oversold
            if oversold and at_lower_donch and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Overbought + Upper Donchian + Volume ===
            if overbought and at_upper_donch and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR for mean reversion = wider, gives room) ===
        if in_position and position_side > 0:
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD: minimum 3 bars (~18h) to avoid chop ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            # Exit when %R reverts to neutral (between -20 and -80)
            mid_willr = -50.0
            if position_side > 0 and willr_14[i] > mid_willr:
                desired_signal = 0.0
            if position_side < 0 and willr_14[i] < mid_willr:
                desired_signal = 0.0
            
            # OR exit when price reaches Donchian middle (mean reversion target)
            if position_side > 0 and close[i] >= donch_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= donch_mid[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals