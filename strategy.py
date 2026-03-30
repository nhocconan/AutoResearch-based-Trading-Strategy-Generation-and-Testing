#!/usr/bin/env python3
"""
Experiment #009: 4h Williams%R + Donchian Breakout + Volume

HYPOTHESIS: Williams %R oscillates [0,-100], making -80/+20 thresholds reliable
for identifying oversold/overbought extremes. Combining with Donchian(20) breakout
(structural levels) and volume confirmation captures mean-reversion trades at
key support/resistance. 1d EMA200 filters trend direction.

WHY IT WORKS: 
- Williams %R is bounded, unlike RSI - thresholds are stable
- Donchian breakout ensures we're at structural support/resistance
- Volume confirms institutional interest at extremes
- Works in BOTH bull (reversals from oversold) and bear (reversals from overbought)

TARGET: 75-200 total 4h trades over 4 years. HARD MAX: 400.
Signal size: 0.25-0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_willr_donchian_vol_1d_v1"
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
    """Williams %R - bounded oscillator [0, -100]"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    willr = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            willr[i] = -50  # neutral when range is zero
    
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend (faster than 50 to avoid missing bear regime)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_14 = calculate_williams_r(high, low, close, period=14)
    
    # Donchian channels (20 periods = 5 days at 4h)
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    mid_band = (upper_band + lower_band) / 2
    
    # Volume ratio (20 period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    warmup = 250  # Need enough for EMA200 alignment buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Williams %R not ready
        if np.isnan(willr_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if 1d EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        bull_trend = close[i] > ema_1d_aligned[i]
        bear_trend = close[i] < ema_1d_aligned[i]
        
        # === MOMENTUM (Williams %R) ===
        # Long when oversold: willr < -80 (extreme)
        # Short when overbought: willr > -20 (extreme)
        oversold = willr_14[i] < -80
        overbought = willr_14[i] > -20
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (structural level) ===
        # Long: price breaks above upper band (new high)
        # Short: price breaks below lower band (new low)
        donchian_breakout_up = close[i] > upper_band[i]
        donchian_breakout_down = close[i] < lower_band[i]
        
        # === MID BAND MEAN REVERSION ===
        # Price too far below mid in bull = potential bounce
        # Price too far above mid in bear = potential drop
        below_mid = close[i] < mid_band[i]
        above_mid = close[i] > mid_band[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Donchian breakout UP + oversold Williams %R + volume + bull trend
            # OR: Price below mid band + oversold + volume + bull trend (mean reversion)
            if (bull_trend and oversold and vol_spike and 
                (donchian_breakout_up or below_mid)):
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Donchian breakout DOWN + overbought Williams %R + volume + bear trend
            # OR: Price above mid band + overbought + volume + bear trend (mean reversion)
            if (bear_trend and overbought and vol_spike and
                (donchian_breakout_down or above_mid)):
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
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
        
        # === TAKE PROFIT: mean reversion to mid band ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 2:
            # Long: take profit when price returns to mid band
            if position_side > 0 and close[i] >= mid_band[i]:
                desired_signal = 0.0
            # Short: take profit when price returns to mid band
            if position_side < 0 and close[i] <= mid_band[i]:
                desired_signal = 0.0
        
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