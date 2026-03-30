#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + 1d EMA50 + Volume Spike

HYPOTHESIS: 12h timeframe with 1d EMA50 trend filter creates a robust
momentum strategy that works across bull, bear, and range markets.
EMA50(1d) alignment is less restrictive than EMA200 while still capturing
major trend direction.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Break above Donchian(20) high + above 1d EMA50 = momentum continuation
- Bear: Break below Donchian(20) low + below 1d EMA50 = momentum continuation
- Range (CHOP > 61.8): Skip entirely, avoid whipsaws
- EMA50 filter prevents fighting major trends

EXPECTED TRADES: 75-150 total over 4 years (19-37/year)
- Donchian(20) on 12h = 10-day channel breaks every 20-40 bars
- ~730 bars/year → 18-36 potential breakouts/year
- Volume spike (1.5x) + EMA50 filter → reduces by ~40%
- CHOP < 50 → reduces by ~30%
- Final: ~15-25/year = 60-100 over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_ema_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: CHOP > 61.8 = range, CHOP < 50 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = sum([
                max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                for j in range(i-period+1, i+1)
            ])
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(ema50_aligned[i]) or np.isnan(ema50_aligned[i-1]):
            signals[i] = 0.0
            continue
        
        # === Regime check ===
        is_trending = chop[i] < 50.0
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # 1d EMA50 trend alignment
        bull_trend = close[i] > ema50_aligned[i]
        bear_trend = close[i] < ema50_aligned[i]
        
        # EMA50 direction confirmation (must be rising/falling)
        ema_rising = ema50_aligned[i] > ema50_aligned[i-1]
        ema_falling = ema50_aligned[i] < ema50_aligned[i-1]
        
        desired_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        if not in_position:
            # LONG: Break above previous Donchian high + volume + trend alignment
            prev_high = donchian_upper[i-1] if i > 0 else np.nan
            long_breakout = (not np.isnan(prev_high)) and (high[i] > prev_high)
            long_conditions = long_breakout and vol_spike and bull_trend and ema_rising and is_trending
            
            if long_conditions:
                desired_signal = SIZE
            else:
                # SHORT: Break below previous Donchian low + volume + trend alignment
                prev_low = donchian_lower[i-1] if i > 0 else np.nan
                short_breakout = (not np.isnan(prev_low)) and (low[i] < prev_low)
                short_conditions = short_breakout and vol_spike and bear_trend and ema_falling and is_trending
                
                if short_conditions:
                    desired_signal = -SIZE
        
        # === EXIT CONDITIONS ===
        if in_position:
            # Exit on EMA50 crossover (trend reversal)
            if position_side > 0:
                if close[i] < ema50_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            elif position_side < 0:
                if close[i] > ema50_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 6 bars (3 days) to avoid fee churn ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        
        signals[i] = desired_signal
    
    return signals