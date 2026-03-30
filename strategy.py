#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian Breakout + Williams %R Regime + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts capture momentum shifts. Using Williams %R
as regime filter (oversold = bullish, overbought = bearish) avoids whipsaws in
ranging markets. Volume confirmation filters false breakouts.

WHY 4h: Optimal trade frequency (20-50/year) with institutional-level structure.
12h strategies in this session consistently underperformed.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Price breaks above Donchian high + Williams %R shows oversold pullback
- Bear: Price breaks below Donchian low + Williams %R shows overbought bounce
- Choppiness regime avoids entries in sideways markets

TARGET: 75-200 total trades over 4 years (19-50/year). HARD MAX: 400.
Signal size: 0.25.

Based on DB winner: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (test Sharpe 1.38)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_willr_vol_1d_v1"
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
    """Williams %R - momentum indicator"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window_high = np.max(high[i - period + 1:i + 1])
        window_low = np.min(low[i - period + 1:i + 1])
        if window_high != window_low:
            willr[i] = -100 * (window_high - close[i]) / (window_high - window_low)
    
    return willr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout structure"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA for trend (simple, proven)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels
    dc_upper_20, dc_mid_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Williams %R
    willr = calculate_williams_r(high, low, close, period=14)
    
    # Volume ratio
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
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME: Williams %R (0 to -100, oversold < -80, overbought > -20)
        willr_val = willr[i] if not np.isnan(willr[i]) else -50
        
        # === TREND: 1d SMA alignment ===
        price_above_sma = close[i] > sma_1d_aligned[i]
        price_below_sma = close[i] < sma_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.4
        
        # Donchian breakout signals (previous bar closes beyond channel)
        upper_broken = close[i - 1] > dc_upper_20[i - 1]  # Previous bar closed above
        lower_broken = close[i - 1] < dc_lower_20[i - 1]  # Previous bar closed below
        
        # Current bar testing breakout level
        testing_upper = high[i] >= dc_upper_20[i - 1]
        testing_lower = low[i] <= dc_lower_20[i - 1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above + oversold pullback (Williams %R < -60)
            # Price above SMA, breakout confirmed, Williams showing pullback
            if price_above_sma and (upper_broken or testing_upper) and vol_spike:
                # Williams %R in oversold zone = good entry
                if willr_val < -60:
                    desired_signal = SIZE
                # OR strong momentum (Williams crossing up from oversold)
                elif i > 0 and willr[i - 1] < -80 and willr_val > -80:
                    desired_signal = SIZE
            
            # === SHORT: Breakout below + overbought bounce (Williams %R > -40)
            if price_below_sma and (lower_broken or testing_lower) and vol_spike:
                # Williams %R in overbought zone = good entry
                if willr_val > -40:
                    desired_signal = -SIZE
                # OR strong momentum (Williams crossing down from overbought)
                elif i > 0 and willr[i - 1] > -20 and willr_val < -20:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT: Exit at 2.5R or DC mid-line ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:
            if position_side > 0:
                profit_target = entry_price + 2.5 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = 0.0
                # Or DC mid-line resistance
                elif close[i] >= dc_mid_20[i]:
                    desired_signal = 0.0
            elif position_side < 0:
                profit_target = entry_price - 2.5 * entry_atr
                if close[i] <= profit_target:
                    desired_signal = 0.0
                # Or DC mid-line support
                elif close[i] <= dc_mid_20[i]:
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals