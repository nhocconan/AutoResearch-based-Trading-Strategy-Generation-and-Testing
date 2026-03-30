#!/usr/bin/env python3
"""
Experiment #025: 12h Donchian Breakout + Choppiness Regime + Volume

HYPOTHESIS: 12h timeframe with tight Donchian(24) + Choppiness regime filter
will generate 50-80 trades over 4 years with positive Sharpe.

WHY IT SHOULD WORK:
- Choppiness Index filters out 50% of bars (choppy = no entries)
- Only enters during trending phases (<38.2 chop = strong trend)
- 12h is slow enough to avoid overtrading but fast enough for statistical validity
- Volume spike confirms momentum, not noise

EXPECTED TRADES: 50-80 total over 4 years (~12-20/year per symbol)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_vol_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (avoid)
    CHOP < 38.2 = trending market (enter)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(period):
            tr = max(high[i-j] - low[i-j], 
                    abs(high[i-j] - close[i-j-1]) if i-j > 0 else high[i-j] - low[i-j],
                    abs(low[i-j] - close[i-j-1]) if i-j > 0 else 0)
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = max(high[i-period+1:i+1])
        ll = min(low[i-period+1:i+1])
        
        if hh - ll > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA for trend direction
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(24) - medium-term structure
    donchian_upper = pd.Series(high).rolling(window=24, min_periods=24).max().values
    donchian_lower = pd.Series(low).rolling(window=24, min_periods=24).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Choppiness Index
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Enough for Donchian24, ATR14, CHOP14
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(daily_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: Choppiness Index ===
        # Only enter when trending (CHOP < 38.2), avoid choppy markets
        is_trending = chop[i] < 38.2
        
        # === TREND DIRECTION: Daily EMA alignment ===
        daily_trend_up = close[i] > daily_ema_aligned[i]
        daily_trend_down = close[i] < daily_ema_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend + trending regime
            if bullish_breakout and vol_spike and daily_trend_up and is_trending:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend + trending regime
            elif bearish_breakout and vol_spike and daily_trend_down and is_trending:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips
                elif close[i] < daily_ema_aligned[i] * 0.99:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips
                elif close[i] > daily_ema_aligned[i] * 1.01:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars (2 days) to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals