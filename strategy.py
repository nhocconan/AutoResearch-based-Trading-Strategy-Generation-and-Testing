#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian(30) + Weekly Trend + Volume Spike

HYPOTHESIS: 1d strategies fail with too few trades because they use short
Donchian periods. By using Donchian(30), we naturally get 2-3 breakouts/month
= 100-144 over 4 years (target range). Combined with weekly trend filter
and volume spike confirmation, this should generate quality signals.

WHY IT WORKS:
- Donchian(30) = 1-month high/low breakout - captures major moves
- Weekly EMA(21) trend = smooths daily noise, confirms direction
- Volume spike (>2x 20d MA) = institutional involvement
- 1d TF = fewer signals = less fee drag = better Sharpe on test
- ATR stoploss adapts to volatility regime

EXPECTED: 80-120 total trades over 4 years (20-30/year)
SIZE: 0.30 (discrete, survives 2022 crash with <25% drawdown)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian30_weekly_trend_v1"
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

def calculate_donchian(high, low, period=30):
    """Donchian Channel - longer period for 1d"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=30)
    
    # Volume ratio (20-period MA for daily = ~1 month)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Precompute daily EMA(50) for medium-term trend ===
    ema_50_daily = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Need 30 for Donchian + 50 for EMA50 + 20 for vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50_daily[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION ===
        # Weekly EMA: HTF trend
        weekly_trend_up = close[i] > ema_aligned[i]
        weekly_trend_down = close[i] < ema_aligned[i]
        
        # Daily EMA(50): medium-term trend (confluence with weekly)
        daily_trend_up = close[i] > ema_50_daily[i]
        daily_trend_down = close[i] < ema_50_daily[i]
        
        # Strong trend = both weekly and daily agree
        strong_uptrend = weekly_trend_up and daily_trend_up
        strong_downtrend = weekly_trend_down and daily_trend_down
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 2.0  # Require 2x average volume
        
        # === DONCHIAN(30) BREAKOUT ===
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        # Breakout = close above/below previous period's channel
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Strong trend + breakout up + volume spike ===
            if breakout_up and strong_uptrend and vol_spike:
                desired_signal = SIZE
            
            # === LONG: Strong trend + breakout up (no volume, but very clean signal) ===
            elif breakout_up and strong_uptrend and close[i] > donchian_up[i] * 1.01:
                desired_signal = SIZE * 0.8  # Smaller without volume confirm
            
            # === SHORT: Strong trend + breakout down + volume spike ===
            if breakout_down and strong_downtrend and vol_spike:
                desired_signal = -SIZE
            
            # === SHORT: Strong trend + breakout down (no volume, but very clean signal) ===
            elif breakout_down and strong_downtrend and close[i] < donchian_lo[i] * 0.99:
                desired_signal = -SIZE * 0.8
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update highest high since entry
                if i == entry_bar or high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                
                # Trailing stop: 3x ATR
                stop_price = highest_since_entry - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips
                if weekly_trend_down:
                    desired_signal = 0.0
                
                # Exit if price drops below daily EMA(50) significantly
                if close[i] < ema_50_daily[i] * 0.97:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update lowest low since entry
                if i == entry_bar or low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                
                # Trailing stop: 3x ATR
                stop_price = lowest_since_entry + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips
                if weekly_trend_up:
                    desired_signal = 0.0
                
                # Exit if price rises above daily EMA(50) significantly
                if close[i] > ema_50_daily[i] * 1.03:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars (avoid fee churn on false breakouts) ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals