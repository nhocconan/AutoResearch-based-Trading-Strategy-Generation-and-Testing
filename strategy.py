#!/usr/bin/env python3
"""
Experiment #028: 6h Donchian Breakout + Williams %R + Volume + 1d SMA50 Trend

HYPOTHESIS: Donchian breakout provides institutional-level price structure,
Williams %R adds momentum confirmation (avoid entries when overbought/oversold 
 extremes reverse), and 1d SMA50 provides directional bias. This combination
 should capture medium-term swings while avoiding false breakouts.

WHY IT WORKS IN BULL AND BEAR:
- Long entries work in bull when price breaks out above SMA50
- Short entries work in bear when price breaks down below SMA50
- Williams %R filters out reversals from extreme levels
- Volume confirms institutional participation
- Choppiness keeps us out of range-bound whipsaws

TARGET: 75-200 total trades over 4 years (19-50/year). HARD MAX: 300.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_williams_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    williams_r = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return williams_r

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 6h indicators (pre-computed for speed)
    atr_14 = pd.Series(close).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # ATR via High/Low/Close method (vectorized)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.vstack([tr1, tr2, tr3])
    tr[0, 0] = tr1[0]  # First bar has no prior close
    tr_max = np.nanmax(tr, axis=0)
    tr_max[0] = tr1[0]
    atr_14 = pd.Series(tr_max).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Choppiness Index (vectorized for speed)
    chop_period = 14
    chop = np.full(n, np.nan, dtype=np.float64)
    tr_sum_arr = np.nancumsum(tr_max)
    
    for i in range(chop_period, n):
        tr_sum = tr_sum_arr[i] - tr_sum_arr[i - chop_period]
        if i - chop_period > 0:
            tr_sum = tr_sum - tr_max[i - chop_period]
        
        hh = np.max(high[i - chop_period + 1:i + 1])
        ll = np.min(low[i - chop_period + 1:i + 1])
        range_hl = hh - ll
        
        if tr_sum > 0 and range_hl > 0:
            chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(chop_period)
    
    # Williams %R (14 period)
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Donchian channels (20 periods = ~5 days on 6h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if critical indicators not ready
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
        
        if np.isnan(chop[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness Index) ===
        # Skip NEW entries when market is too choppy (CHOP > 61.8)
        is_choppy = chop[i] > 61.8
        
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === MOMENTUM CHECK (Williams %R) ===
        # Avoid entering when momentum is extremely stretched
        # Long: williams_r should be > -80 (not deeply oversold which often reverses)
        # Short: williams_r should be < -20 (not deeply overbought which often reverses)
        willr = williams_r[i]
        momentum_ok_long = willr > -95  # Not deeply oversold
        momentum_ok_short = willr < -5   # Not deeply overbought
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        current_high = high[i]
        current_low = low[i]
        
        # Previous bar's Donchian values (proper alignment, no lookahead)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above previous Donchian high ===
            # Price breaks above 20-bar high with trend, momentum, and volume confirmation
            if current_high > prev_donchian_high and price_above_1d_sma:
                if vol_spike and momentum_ok_long:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below previous Donchian low ===
            # Price breaks below 20-bar low with trend, momentum, and volume confirmation
            if current_low < prev_donchian_low and not price_above_1d_sma:
                if vol_spike and momentum_ok_short:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 6 bars = 1.5 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 6:
            # Exit if price reverts to middle of channel
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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