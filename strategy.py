#!/usr/bin/env python3
"""
Experiment #011: 6h Elder Ray + 1d Trend + Choppiness Regime

HYPOTHESIS: Elder Ray (Bull/Bear power) measures institutional buying/selling pressure
relative to a smoothed moving average. Combined with 1d SMA200 for trend direction and
Choppiness Index to filter out range-bound periods, this captures mean-reversion moves
at key turning points. Elder Ray is particularly effective in volatile markets like 2022
where it captures the institutional footprint during large swings.

WHY 6h: Slower than 4h (reduces fee drag), faster than 12h (more opportunities).
Elder Ray works well on this TF because institutional moves take 6-12h to develop.

KEY DIFFERENCE FROM PREVIOUS: Not another Donchian/HMA crossover. Elder Ray measures
the "shadow" of price action relative to EMA - capturing where institutions are
active vs passive. Combined with Choppiness to stay out of range-bound markets.

TARGET: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 300.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1d_sma200_chop_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (momentum works)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            # Highest high - lowest low over period
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                # CHOP = 100 * log10(tr_sum / range_hl) / log10(period)
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_elder_ray(high, low, close, ema_period=13):
    """
    Elder Ray (Bull Power / Bear Power)
    Bull Power = High - EMA(13)
    Bear Power = Low - EMA(13)
    Oscillator = Bull Power + Bear Power (or just use Bull Power)
    """
    ema = calculate_ema(close, ema_period)
    
    n = len(close)
    bull_power = high - ema
    bear_power = low - ema
    
    return bull_power, bear_power, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # 1d ATR for stoploss sizing
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    bull_power, bear_power, ema_13 = calculate_elder_ray(high, low, close, ema_period=13)
    
    # Elder Ray smoothed (EMA of the power values)
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # CHOP < 38.2 = trending (Elder Ray momentum works)
        # CHOP > 61.8 = choppy (Elder Ray mean reversion works)
        is_trending = chop[i] < 38.2
        is_choppy = chop[i] > 61.8
        
        # === ELDER RAY SIGNALS ===
        bull = bull_power_smooth[i]
        bear = bear_power_smooth[i]
        
        # Normalize Elder Ray by ATR for comparison
        atr_local = atr_14[i]
        bull_norm = bull / atr_local if atr_local > 0 else 0
        bear_norm = bear / atr_local if atr_local > 0 else 0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === TRENDING MARKET: Follow momentum ===
            if is_trending:
                # Long: Bull Power > 0 AND price above 1d SMA
                if bull_norm > 0.3 and price_above_1d_sma:
                    if vol_spike:
                        desired_signal = SIZE
                
                # Short: Bear Power < 0 AND price below 1d SMA
                if bear_norm < -0.3 and not price_above_1d_sma:
                    if vol_spike:
                        desired_signal = -SIZE
            
            # === CHOPPY MARKET: Fade extremes ===
            if is_choppy:
                # Long: Bear Power recovering from negative (buying pressure)
                # Bear crossed from below -0.5*ATR to above it
                bear_prev = bear_power_smooth[i-1] if i > 1 else 0
                bear_prev_norm = bear_prev / atr_local if atr_local > 0 else 0
                
                if bear_norm > -0.2 and bear_prev_norm < -0.4 and price_above_1d_sma:
                    desired_signal = SIZE
                
                # Short: Bull Power collapsing from positive (selling pressure)
                bull_prev = bull_power_smooth[i-1] if i > 1 else 0
                bull_prev_norm = bull_prev / atr_local if atr_local > 0 else 0
                
                if bull_norm < 0.2 and bull_prev_norm > 0.4 and not price_above_1d_sma:
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
        
        # === TIME-BASED EXIT (hold at least 4 bars = 1 day) ===
        bars_held = i - (entry_bar if in_position else i)
        min_hold_bars = 4
        
        if in_position and bars_held >= min_hold_bars:
            # Exit if Elder Ray reverses
            if position_side > 0 and bear_norm < -0.3:
                desired_signal = 0.0
            if position_side < 0 and bull_norm > 0.3:
                desired_signal = 0.0
        
        # === RSI EXIT FILTER ===
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = (100 - (100 / (1 + rs)))[i]
        
        if in_position:
            if position_side > 0 and rsi > 75:
                desired_signal = 0.0
            if position_side < 0 and rsi < 25:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        entry_bar = i  # Track for time-based exit
        
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
                # Same direction - maintain position
                pass
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