#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + Volume + 1d Trend

HYPOTHESIS: Donchian(20) on 12h captures multi-day swing breakouts.
Volume confirmation filters false breakouts. 1d SMA200 ensures trend alignment.
Simple = fewer trades = less fee drag = better generalization.

WHY 12h: ~292 12h bars/year. Target 15-30 trades/year = 5% trigger rate.
Keep it simple: 2 conditions (breakout + volume), trend as filter.

FOLLOWING DB WINNER PATTERN: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1
(SOL: test_sharpe=1.38, 95 trades) — adapted to 12h for fewer trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_sma200_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend (proven in DB winners)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channel (20 periods = 10 days)
    donch_period = 20
    donch_high = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    donch_low = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # 12h SMA for momentum direction
    sma_12h = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    
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
    
    warmup = max(250, donch_period)  # Need 200 for SMA200 + 20 for Donchian
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND FILTER (1d SMA200) ===
        bull_trend = close[i] > sma_1d_aligned[i]
        bear_trend = close[i] < sma_1d_aligned[i]
        
        # === MOMENTUM (12h SMA direction) ===
        sma_rising = sma_12h[i] > sma_12h[i - 1] if i > warmup else False
        sma_falling = sma_12h[i] < sma_12h[i - 1] if i > warmup else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === DONCHIAN BREAKOUT (using PREVIOUS bar's Donchian = no look-ahead) ===
        prev_donch_high = donch_high[i - 1]
        prev_donch_low = donch_low[i - 1]
        
        # Upper breakout: close above previous 20-bar high
        upper_breakout = close[i] > prev_donch_high
        # Lower breakout: close below previous 20-bar low
        lower_breakout = close[i] < prev_donch_low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Upper Donchian breakout + bull trend + volume ===
            if upper_breakout and bull_trend and vol_spike:
                desired_signal = SIZE
            # Fallback: upper breakout + rising SMA (less strict)
            elif upper_breakout and sma_rising and vol_spike:
                desired_signal = SIZE * 0.5  # Half size without trend confirmation
            
            # === SHORT: Lower Donchian breakout + bear trend + volume ===
            if lower_breakout and bear_trend and vol_spike:
                desired_signal = -SIZE
            # Fallback: lower breakout + falling SMA (less strict)
            elif lower_breakout and sma_falling and vol_spike:
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT: trail stop after 2R profit ===
        if in_position:
            bars_held = i - entry_bar
            if bars_held >= 4:  # Minimum hold 2 days
                if position_side > 0:
                    profit = (high[i] - entry_price) / entry_atr
                    if profit >= 3.0:
                        # Lock in profits: trail stop tighter
                        stop_price = highest_since_entry - 1.5 * entry_atr
                        if low[i] < stop_price:
                            desired_signal = 0.0
                elif position_side < 0:
                    profit = (entry_price - low[i]) / entry_atr
                    if profit >= 3.0:
                        stop_price = lowest_since_entry + 1.5 * entry_atr
                        if high[i] > stop_price:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals