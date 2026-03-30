#!/usr/bin/env python3
"""
Experiment #023: 12h Donchian Breakout + Volume Spike + 1d Trend (12h)

HYPOTHESIS: Price channel breakout on 12h with volume confirmation and
1d EMA50 trend filter provides robust entries across bull/bear cycles.

WHY 12h WORKS:
- 12h = 2920 bars/year → fewer signals = less fee drag
- Donchian(20) on 12h = structural breaks every ~240 bars (5-10/year per direction)
- Volume spike + trend filter reduces false breakouts
- ATR trailing stop manages risk without whipsaw

ENTRY LOGIC (simple, 2 conditions):
1. Price breaks Donchian(20) high/low on 12h
2. Volume spike (2x 20-bar MA) confirms breakout

EXIT LOGIC:
- ATR(14) * 2.5 trailing stop from entry
- Exit on opposite Donchian touch OR ATR-based stop

FILTERS:
- 1d EMA50 aligned to 12h for trend direction
- No trade if price and EMA50 disagree on direction (confluence required)

EXPECTED TRADE COUNT: 60-120 total over 4 years (15-30/year)
- Donchian breaks ~10-20 times/year (long + short potential)
- Volume filter (2x) → reduces by ~30%
- Trend filter (1d EMA50) → reduces by ~40%
- Final: ~12-25 trades/year = 48-100 total over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_ema50_v1"
timeframe = "12h"
leverage = 1.0

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
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF indicators (1d) ===
    # 1d EMA50 for trend direction
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) on 12h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
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
    
    warmup = 30  # Donchian(20) needs 20 bars, ATR(14) needs 14
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i-1]) or np.isnan(donchian_lower[i-1]):
            signals[i] = 0.0
            continue
        
        # Get 1d trend
        htf_trend_up = ema50_1d_aligned[i] > close[i] if not np.isnan(ema50_1d_aligned[i]) else False
        htf_trend_down = ema50_1d_aligned[i] < close[i] if not np.isnan(ema50_1d_aligned[i]) else False
        
        # Volume spike: 2x above average
        vol_spike = vol_ratio[i] > 2.0
        
        # Donchian breakout detection (use PREVIOUS bar's channel - no look-ahead)
        bullish_breakout = high[i] > donchian_upper[i-1]
        bearish_breakout = low[i] < donchian_lower[i-1]
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: Break above Donchian high + volume spike + 1d trend up
            if bullish_breakout and vol_spike and htf_trend_up:
                desired_signal = SIZE
            
            # SHORT: Break below Donchian low + volume spike + 1d trend down
            elif bearish_breakout and vol_spike and htf_trend_down:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Update highest since entry
                if high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                
                # ATR trailing stop: 2.5 ATR from highest
                stop_dist = 2.5 * entry_atr
                if atr_14[i] > 0:
                    stop_dist = min(stop_dist, 3.0 * atr_14[i])  # Cap at 3 ATR
                stop_price = highest_since_entry - stop_dist
                
                # Stopped out
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                else:
                    desired_signal = SIZE
                    
            elif position_side < 0:
                # Update lowest since entry
                if low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                
                # ATR trailing stop: 2.5 ATR from lowest
                stop_dist = 2.5 * entry_atr
                if atr_14[i] > 0:
                    stop_dist = min(stop_dist, 3.0 * atr_14[i])
                stop_price = lowest_since_entry + stop_dist
                
                # Stopped out
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                else:
                    desired_signal = -SIZE
        
        # === TAKE PROFIT AT 3R ===
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                # Take partial profit, keep half position
                desired_signal = SIZE / 2
                
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                desired_signal = -SIZE / 2
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        
        signals[i] = desired_signal
    
    return signals