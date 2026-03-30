#!/usr/bin/env python3
"""
Experiment #023: 4h Donchian Breakout + Volume + 1d Trend + ATR Distance

HYPOTHESIS: Simple price-channel breakout (Donchian 20) with strict ATR-distance
filtering naturally prevents overtrading. The ATR distance from recent extremes
acts as a volatility-normalized confirmation that reduces false breakouts.
Combined with 1d SMA200 for trend direction and volume confirmation.

WHY 4h: Proven in DB - best test Sharpe strategies use 4h. Fast enough for
meaningful trades (50-150/year), slow enough to avoid fee drag.

WHY SIMPLE: Complex strategies (Elder Ray, Williams %R, Alligator) all failed.
The winning strategies use price channels + volume + regime. Nothing more.

TRADE COUNT TARGET: 75-150 total over 4 years (19-37/year). HARD MAX: 300.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_atr_dist_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range using EWM"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Pre-compute indicators
    donchian_upper = np.zeros(n)
    donchian_lower = np.zeros(n)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20+1:i+1])
        donchian_lower[i] = np.min(low[i-20+1:i+1])
    
    atr_14 = calculate_atr(high, low, close, period=14)
    
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    chop = calculate_choppiness(high, low, close, period=14)
    
    # HTF: 1d SMA200 for trend
    df_1d = get_htf_data(prices, '1d')
    sma_200 = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    
    # RSI for exit filter
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Signals
    signals = np.zeros(n)
    
    # Position state
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    SIZE = 0.30
    
    warmup = 250
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            position_side = 0
            continue
            
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        desired_signal = 0.0
        atr_dist = atr_14[i] * 1.5
        
        # Trend direction
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        
        # CHOP regime
        is_choppy = not np.isnan(chop[i]) and chop[i] > 61.8
        
        # Extra ATR distance for choppy markets
        chop_extra = atr_14[i] * 1.5 if is_choppy else 0.0
        
        # ========== OPENING NEW POSITION ==========
        if position_side == 0:
            # LONG: Breakout above upper Donchian + ATR distance + volume + trend
            if close[i] > donchian_upper[i] + atr_dist + chop_extra:
                if vol_ratio[i] > 1.5:
                    if price_above_1d_sma:
                        desired_signal = SIZE
            
            # SHORT: Breakdown below lower Donchian + ATR distance + volume + trend
            elif close[i] < donchian_lower[i] - atr_dist - chop_extra:
                if vol_ratio[i] > 1.5:
                    if not price_above_1d_sma:
                        desired_signal = -SIZE
        
        # ========== TRAILING STOP (after min hold) ==========
        else:
            bars_held = i - entry_bar
            
            # Update high/low
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                if low[i] < trailing_stop:
                    desired_signal = 0.0
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    entry_bar = 0
                    
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                if high[i] > trailing_stop:
                    desired_signal = 0.0
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    entry_bar = 0
            
            # RSI exit filter (after min hold)
            if position_side != 0 and bars_held >= 6:
                rsi_val = rsi[i]
                if not np.isnan(rsi_val):
                    if position_side > 0 and rsi_val > 75:
                        desired_signal = 0.0
                        position_side = 0
                    elif position_side < 0 and rsi_val < 25:
                        desired_signal = 0.0
                        position_side = 0
                
                if position_side == 0:
                    entry_price = 0.0
                    entry_atr = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    entry_bar = 0
            
            # Flip direction if signal reversed
            if position_side != 0 and desired_signal != 0:
                if np.sign(desired_signal) != position_side:
                    position_side = int(np.sign(desired_signal))
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    entry_bar = i
        
        # ========== OPEN NEW POSITION ==========
        if desired_signal != 0.0 and position_side == 0:
            position_side = int(np.sign(desired_signal))
            entry_price = close[i]
            entry_atr = atr_14[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            entry_bar = i
        
        signals[i] = desired_signal if position_side != 0 else 0.0
    
    return signals