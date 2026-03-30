#!/usr/bin/env python3
"""
Experiment #028: 4h TRIX Momentum + Donchian Structure + 12h Trend + Volume Spike

HYPOTHESIS: TRIX (Triple EMA) filters market noise better than single EMA and catches
momentum shifts earlier. Combined with:
- Donchian(25) for price structure (slightly longer = fewer false breakouts)
- 12h SMA50 for trend (faster than 1d = more responsive but still filtered)
- Volume surge (2.0x) for institutional confirmation
- ATR(14) stoploss at 2.5x (wider = room to breathe, fewer premature stops)

WHY IT WORKS: TRIX crossover is a LEADING momentum signal, not lagging like price
breakout alone. When TRIX crosses zero WITH a price structure breakout AND volume surge,
it's a high-probability move. The 12h trend filter avoids fighting the larger trend.

TARGET: 60-150 total trades over 4 years = 15-37/year.
Signal size: 0.30 (moderate).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_donchian_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_trix(close, period=14):
    """Triple EMA Oscillator - leading momentum signal"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA smoothing
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # TRIX = rate of change of triple EMA
    trix = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if ema3.iloc[i-1] != 0:
            trix[i] = ((ema3.iloc[i] - ema3.iloc[i-1]) / ema3.iloc[i-1]) * 100
    
    return trix

def calculate_donchian(high, low, period=20):
    """Donchian Channels for structure"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h SMA50 for trend direction
    sma_12h = pd.Series(df_12h['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_12h)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    trix = calculate_trix(close, period=14)
    
    # Donchian(25) for structure - slightly longer = fewer false breakouts
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=25)
    
    # Volume - use stronger filter (2.0x) to reduce false breakouts
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 150  # Need enough for TRIX + Donchian(25) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(trix[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (12h SMA50) ===
        price_above_12h_sma = close[i] > sma_12h_aligned[i]
        price_below_12h_sma = close[i] < sma_12h_aligned[i]
        
        # === TRIX MOMENTUM ===
        # TRIX > 0 = bullish momentum, TRIX < 0 = bearish momentum
        trix_bullish = trix[i] > 0
        trix_bearish = trix[i] < 0
        
        # === DONCHIAN STRUCTURE ===
        # Use PREVIOUS bar's Donchian to avoid look-ahead
        prev_donchian_upper = donchian_upper[i - 1] if i > 0 else 0
        prev_donchian_lower = donchian_lower[i - 1] if i > 0 else 0
        
        # === VOLUME CONFIRMATION (2.0x - stronger filter) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Requirements: 
            # 1. Price breaks above previous Donchian high
            # 2. 12h trend is bullish
            # 3. TRIX shows bullish momentum
            # 4. Volume surge confirms (optional but helps)
            breakout_long = high[i] > prev_donchian_upper
            trend_long = price_above_12h_sma
            momentum_long = trix_bullish
            
            if breakout_long and trend_long and momentum_long:
                if vol_spike:  # Strong confirmation
                    desired_signal = SIZE
                elif i - (entry_bar if entry_bar > 0 else 0) > 10:  # Or wait 10 bars if no vol
                    desired_signal = SIZE * 0.5  # Half size without volume confirmation
            
            # === SHORT ENTRY ===
            # Requirements:
            # 1. Price breaks below previous Donchian low
            # 2. 12h trend is bearish
            # 3. TRIX shows bearish momentum
            # 4. Volume surge confirms
            breakout_short = low[i] < prev_donchian_lower
            trend_short = price_below_12h_sma
            momentum_short = trix_bearish
            
            if breakout_short and trend_short and momentum_short:
                if vol_spike:
                    desired_signal = -SIZE
                elif i - (entry_bar if entry_bar > 0 else 0) > 10:
                    desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days on 4h) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit if price reverts to Donchian mid or momentum flips
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
                desired_signal = 0.0
            
            # Or exit if momentum fully reverses
            if position_side > 0 and trix[i] < -0.5:
                desired_signal = 0.0
            if position_side < 0 and trix[i] > 0.5:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals