#!/usr/bin/env python3
"""
Experiment #019: 12h Donchian Breakout + Weekly EMA Trend + Volume

HYPOTHESIS: Donchian(20) breakout captures momentum shifts. Combining with
1-week EMA trend filter ensures we're trading with the longer-term direction.
Volume confirmation validates institutional participation. ATR stoploss manages risk.

WHY 12h: ~3x fewer trades than 4h = less fee drag. Weekly EMA on 12h bars
captures multi-week trends without noise.

WHY IT WORKS IN BULL AND BEAR: Symmetrical - long breakouts in uptrends,
short breakouts in downtrends. Weekly EMA filter prevents fighting the trend.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.

Based on DB winner: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (test Sharpe 1.382)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_weekly_ema_vol_v1"
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

def calculate_ema(data, span, min_periods=None):
    """Exponential Moving Average"""
    if min_periods is None:
        min_periods = span
    return pd.Series(data).ewm(span=span, min_periods=min_periods, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA21 for trend direction (smoother than daily)
    ema_1w = calculate_ema(df_1w['close'].values, span=21, min_periods=21)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume rolling mean
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channel (20 periods on 12h = 10 days)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Weekly EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Current HTF trend
        weekly_trend_up = close[i] > ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout signals
        upper_touch = donchian_high[i] > 0 and high[i] >= donchian_high[i]
        lower_touch = donchian_low[i] > 0 and low[i] <= donchian_low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Upper Donchian breakout + weekly trend up + volume ===
            if upper_touch and weekly_trend_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Lower Donchian breakout + weekly trend down + volume ===
            if lower_touch and not weekly_trend_up and vol_spike:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position and position_side > 0:
            # Update highest
            highest_since_entry = max(highest_since_entry, high[i])
            
            # Stop if price breaks below weekly EMA (trend reversal)
            if close[i] < ema_1w_aligned[i]:
                desired_signal = 0.0
            
            # ATR trailing stop
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Update lowest
            lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Stop if price breaks above weekly EMA (trend reversal)
            if close[i] > ema_1w_aligned[i]:
                desired_signal = 0.0
            
            # ATR trailing stop
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals