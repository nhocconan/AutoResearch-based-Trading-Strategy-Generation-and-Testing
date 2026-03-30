#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + 1d EMA Trend + Volume

HYPOTHESIS: 12h timeframe = ~3x fewer trades than 4h, reducing fee drag.
Donchian(20) on 12h captures medium-term breakouts (~10-20 bar holds).
1d EMA50 ensures trend alignment (avoids trading against daily trend).
Volume confirmation filters false breakouts.

WHY BOTH BULL AND BEAR: Breakout systems work in both:
- Bull: buy breaks of 20-bar high with price > 1d EMA
- Bear: short breaks of 20-bar low with price < 1d EMA

TARGET: 75-150 total trades over 4 years (19-37/year).
Previous 12h attempts: #013 (6tr), #016 (60tr), #019 (39tr) — too strict or neg Sharpe.
This loosens volume threshold and Donchian period for more signals.

DB PROVEN: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 → test Sharpe 1.38 (SOL)
DB PROVEN: mtf_4h_hma_volume_donchian_adx_12h_atr_v1 → test Sharpe 1.32 (SOL)
This is the 12h version of those proven patterns.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_ema50_vol_1d_v1"
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
    
    # 1d EMA50 for trend direction (must be aligned to avoid look-ahead)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channels (20 bars = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = max(100, donchian_period + 20)  # Need enough for Donchian + volume MA
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        # Current HTF levels
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        
        # Skip if Donchian not ready
        if np.isnan(dc_high) or np.isnan(dc_low):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        is_bull_trend = close[i] > ema_1d_aligned[i]
        is_bear_trend = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation (1.3x above 20-bar average)
        vol_confirm = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Break above 20-bar high with trend + volume ===
            if is_bull_trend and vol_confirm:
                if high[i] > dc_high:
                    desired_signal = SIZE
            
            # === SHORT: Break below 20-bar low with trend + volume ===
            if is_bear_trend and vol_confirm:
                if low[i] < dc_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR from entry) ===
        if in_position:
            bars_held = i - entry_bar
            
            if position_side > 0:
                stop_loss = entry_price - 2.0 * entry_atr
                # Stop if price drops below entry - 2*ATR
                if low[i] < stop_loss:
                    desired_signal = 0.0
                # Also exit if we drop back below Donchian high (failed breakout)
                elif close[i] < dc_high and bars_held >= 3:
                    desired_signal = 0.0
            
            if position_side < 0:
                stop_loss = entry_price + 2.0 * entry_atr
                # Stop if price rises above entry + 2*ATR
                if high[i] > stop_loss:
                    desired_signal = 0.0
                # Also exit if we rise back above Donchian low (failed breakdown)
                elif close[i] > dc_low and bars_held >= 3:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals