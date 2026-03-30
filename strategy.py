#!/usr/bin/env python3
"""
Experiment #021: 12h Williams Alligator + Elder Ray + Volume

HYPOTHESIS: Williams Alligator captures institutional trend cycles.
When price closes outside all 3 lines (Jaw/Teeth/Lips) AND Elder Ray confirms
bull power (for longs) or bear power (for shorts) AND volume spikes, this
indicates strong directional moves. 1d SMA200 adds longer-term structure.

WHY 12h: 3x slower than 4h = fewer but higher quality signals.
Alligator "sleeping" periods filter out choppy markets naturally.
Works in BOTH bull (buy breakout) and bear (short breakout).

TARGET: 75-150 total trades over 4 years = 19-37/year.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_alligator_elder_ray_vol_1d_v1"
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

def calculate_alligator(high, low, close):
    """
    Williams Alligator:
    - Jaw (blue): SMA(13) of median price, shifted 8 bars
    - Teeth (red): SMA(8) of median price, shifted 5 bars
    - Lips (green): SMA(5) of median price, shifted 3 bars
    """
    n = len(close)
    median = (high + low + close) / 3.0
    
    jaw = pd.Series(median).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median).rolling(window=5, min_periods=5).mean().shift(3).values
    
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, ema_period=13):
    """
    Elder Ray:
    - Bull Power = High - EMA(13)
    - Bear Power = Low - EMA(13)
    """
    n = len(close)
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    bull_power = high - ema
    bear_power = low - ema
    
    return bull_power, bear_power

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend structure (shift by 1 to avoid look-ahead)
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().shift(1).values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ema_period=13)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian for breakout confirmation (shift 1)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 150  # Need enough for Alligator alignment + SMA200 buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Alligator or SMA200 not aligned
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === STRUCTURE: 1d SMA200 trend (shifted, so current bar uses PREVIOUS 1d value) ===
        price_above_sma200 = close[i] > sma_200_aligned[i]
        
        # === ALLIGATOR STATE: Check if "mouth" is open or closed ===
        # Bullish: Lips > Teeth > Jaw (lines aligned up)
        # Bearish: Lips < Teeth < Jaw (lines aligned down)
        # Choppy: Lines intertwined
        alligator_bull = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bear = lips[i] < teeth[i] and teeth[i] < jaw[i]
        alligator_open = alligator_bull or alligator_bear
        
        # === ELDER RAY: Bull/Bear power (need shift for EMA look-ahead) ===
        # Bull power should be positive for longs, negative for shorts
        bull_ok = bull_power[i] > 0
        bear_ok = bear_power[i] < 0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_high_20[i]
        donchian_breakout_down = close[i] < donchian_low_20[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # All conditions must align:
            # 1. Price above 1d SMA200 (structural bull)
            # 2. Alligator lines bullish (mouth open up)
            # 3. Bull power positive (buying pressure)
            # 4. Volume spike (institutional)
            # 5. Breakout above 20-bar high
            if (price_above_sma200 and alligator_bull and bull_ok and 
                vol_spike and donchian_breakout_up):
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # 1. Price below 1d SMA200 (structural bear)
            # 2. Alligator lines bearish (mouth open down)
            # 3. Bear power negative (selling pressure)
            # 4. Volume spike (institutional)
            # 5. Breakdown below 20-bar low
            if (not price_above_sma200 and alligator_bear and bear_ok and 
                vol_spike and donchian_breakout_down):
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR) ===
        if in_position and position_side > 0:
            # Trailing stop: highest - 2*ATR
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Trailing stop: lowest + 2*ATR
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars (1.5 days on 12h) to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:
            # Take profit at 2R
            if position_side > 0:
                profit_target = entry_price + 2.0 * entry_atr
                if close[i] >= profit_target:
                    desired_signal = 0.0
            if position_side < 0:
                profit_target = entry_price - 2.0 * entry_atr
                if close[i] <= profit_target:
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
                # Stopped out or took profit
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals