#!/usr/bin/env python3
"""
Experiment #005 variant: 12h Williams %R + Donchian Breakout + 1d SMA Filter

HYPOTHESIS: Williams %R identifies momentum extremes that often reverse in range
or mean-revert within trends. Combined with Donchian 20 channel breakout 
confirmation, this catches the "snap back" moves after false breakouts.
- In 2021 bull: %R oversold bounces when BTC pulls back to SMA(50)
- In 2022 bear: %R overbought in rallies back to SMA(50)
- In 2025 range: %R signals work well at channel boundaries

KEY INSIGHT: Winning 12h strategies use simple price structure (Donchian)
+ ONE momentum filter (RSI, %R, or volume). This keeps trade count in range.

TRADE COUNT: 60-120 total over 4 years (15-30/year).
Size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_donchian_1d_sma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator [-100 to 0]"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 1e-10:
            willr[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            willr[i] = -50.0  # neutral
    
    return willr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20-period breakout"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, middle, lower

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

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA(50) for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === 12h Local indicators ===
    willr_14 = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_mid, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume metrics
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 80  # Donchian(20) + Williams %R
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(willr_14[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Price breaks above 20-period high = bullish breakout
        price_broke_high = close[i] > donchian_upper[i]
        # Price breaks below 20-period low = bearish breakout
        price_broke_low = close[i] < donchian_lower[i]
        
        # Price at or near channel mid = in range (avoid)
        channel_width = donchian_upper[i] - donchian_lower[i]
        price_at_mid_ratio = abs(close[i] - donchian_mid[i]) / channel_width if channel_width > 1e-10 else 0
        in_channel = price_at_mid_ratio < 0.2  # within 20% of mid = range
        
        # === WILLIAMS %R SIGNALS ===
        # %R oversold (< -80) = potential bounce
        willr_oversold = willr_14[i] < -80
        # %R overbought (> -20) = potential reversal
        willr_overbought = willr_14[i] > -20
        # %R neutral zone for exit
        willr_neutral_long = willr_14[i] > -50   # exited oversold
        willr_neutral_short = willr_14[i] < -50  # exited overbought
        
        # === HTF MACRO FILTER ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === MINIMUM HOLD: 2 bars (24h) to avoid whipsaws ===
        min_hold_bars = 2
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_trailing_stop():
            if not in_position:
                return False
            if position_side > 0:
                # Long stop: trail from highest since entry
                highest_since = np.max(high[entry_bar:i+1])
                return low[i] < (highest_since - 2.5 * entry_atr)
            else:
                # Short stop: trail from lowest since entry
                lowest_since = np.min(low[entry_bar:i+1])
                return high[i] > (lowest_since + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_trailing_stop()
            
            # Exit on %R reversal with min hold
            if position_side > 0 and willr_overbought and min_hold:
                stop_hit = True
            if position_side < 0 and willr_oversold and min_hold:
                stop_hit = True
            
            # Exit when price returns to channel mid
            if position_side > 0 and price_at_mid_ratio < 0.1 and min_hold:
                stop_hit = True
            if position_side < 0 and price_at_mid_ratio < 0.1 and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: %R oversold + breakout above channel + HTF bullish + volume
            if willr_oversold and price_broke_high and htf_bullish and vol_confirm:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: %R overbought + breakdown below channel + HTF bearish + volume
            elif willr_overbought and price_broke_low and htf_bearish and vol_confirm:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals