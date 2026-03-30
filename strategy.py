#!/usr/bin/env python3
"""
Experiment #028: 12h Supertrend Breakout + Volume + 1w ATR Regime

HYPOTHESIS: Supertrend(10,3) provides self-adjusting price channel structure
that works in both bull and bear markets. Combined with 1w trend direction
and ATR-based regime filter (avoid high volatility explosions), this captures
sustainable trend changes without overtrading.

WHY 12h: Cuts trade frequency by ~50% vs 6h, targets 12-37 trades/year.
12h = 365 bars/year, so 50 trades = 13.7% trade frequency.

WHY IT WORKS: Supertrend flips direction based on ATR volatility, so it
automatically adapts to trending vs ranging. The 1w EMA(21) gives longer-term
bias. Volume confirms institutional participation. ATR regime filter avoids
entering during vol explosions (often reversal points).

TARGET: 50-120 total trades over 4 years. STRICT filters to avoid overtrading.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_vol_1w_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator - returns upper band, lower band, and direction"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.full(n, np.nan, dtype=np.float64)
    lower_band = np.full(n, np.nan, dtype=np.float64)
    supertrend = np.zeros(n, dtype=np.float64)  # 1 = bullish, -1 = bearish
    
    # HL2 = (high + low) / 2
    hl2 = (high + low) / 2
    
    supertrend_value = hl2[period] - multiplier * atr[period]
    supertrend[period] = 1.0  # Start bullish
    
    for i in range(period, n):
        if i > period:
            bull_band = hl2[i] - multiplier * atr[i]
            bear_band = hl2[i] + multiplier * atr[i]
            
            # Upper band: use previous upper band unless broken
            if not np.isnan(upper_band[i-1]):
                if close[i] > upper_band[i-1]:
                    upper_band[i] = bull_band
                else:
                    upper_band[i] = min(bull_band, upper_band[i-1])
            else:
                upper_band[i] = bull_band
            
            # Lower band: use previous lower band unless broken
            if not np.isnan(lower_band[i-1]):
                if close[i] < lower_band[i-1]:
                    lower_band[i] = bear_band
                else:
                    lower_band[i] = max(bear_band, lower_band[i-1])
            else:
                lower_band[i] = bear_band
            
            # Supertrend direction
            if supertrend[i-1] == 1.0:  # Was bullish
                if close[i] < lower_band[i-1] if not np.isnan(lower_band[i-1]) else close[i] < hl2[i]:
                    supertrend[i] = -1.0
                else:
                    supertrend[i] = 1.0
            else:  # Was bearish
                if close[i] > upper_band[i-1] if not np.isnan(upper_band[i-1]) else close[i] > hl2[i]:
                    supertrend[i] = 1.0
                else:
                    supertrend[i] = -1.0
        else:
            upper_band[i] = hl2[i] - multiplier * atr[i]
            lower_band[i] = hl2[i] + multiplier * atr[i]
            supertrend[i] = 1.0
    
    return upper_band, lower_band, supertrend

def calculate_volatility_ratio(atr, period=10):
    """ATR ratio: current ATR vs recent average - detects vol regime changes"""
    n = len(atr)
    vol_ratio = np.full(n, 1.0, dtype=np.float64)
    
    for i in range(period, n):
        if not np.isnan(atr[i]) and atr[i] > 0:
            atr_avg = np.mean(atr[max(0, i-period):i])
            if atr_avg > 0:
                vol_ratio[i] = atr[i] / atr_avg
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(21) for long-term trend
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    upper, lower, supertrend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    vol_ratio = calculate_volatility_ratio(atr_14, period=10)
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / np.where(vol_sma > 0, vol_sma, 1)
    
    # Williams %R (14) for momentum
    williams_r = np.full(n, np.nan, dtype=np.float64)
    for i in range(14, n):
        highest_high = np.max(high[i-14:i+1])
        lowest_low = np.min(low[i-14:i+1])
        if highest_high > lowest_low:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
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
    
    warmup = 200  # Need enough for Supertrend + 1w alignment + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === LONG-TERM TREND (1w EMA21) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # === VOLATILITY REGIME FILTER ===
        # Skip if volatility is too high (often reversal points)
        # Skip if volatility is too low (choppy, no trend)
        vol_regime_ok = (vol_ratio[i] >= 0.7) and (vol_ratio[i] <= 2.5)
        
        if not vol_regime_ok and not in_position:
            signals[i] = 0.0
            continue
        
        # === VOLUME CONFIRMATION ===
        # Require stronger volume for entry
        volume_confirmed = vol_spike[i] > 1.6
        
        # === MOMENTUM CONFIRMATION ===
        # Williams %R: between -30 and -70 for long entry, between -30 and -70 for short
        momentum_ok = not np.isnan(williams_r[i])
        
        # === SUPERTREND SIGNALS ===
        st_direction = supertrend[i] if not np.isnan(supertrend[i]) else 0
        st_prev = supertrend[i - 1] if i > 0 and not np.isnan(supertrend[i - 1]) else st_direction
        st_flipped = (st_direction != st_prev) and (st_prev != 0)
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Supertrend flips to bullish AND price above 1w EMA (trend confirmation)
            # OR Supertrend already bullish with volume spike
            if price_above_1w_ema:
                if st_flipped and st_direction > 0:
                    if volume_confirmed or momentum_ok:
                        desired_signal = SIZE
                elif st_direction > 0 and volume_confirmed:
                    # Trend continuation with volume
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Supertrend flips to bearish AND price below 1w EMA
            # OR Supertrend already bearish with volume spike
            if not price_above_1w_ema:
                if st_flipped and st_direction < 0:
                    if volume_confirmed or momentum_ok:
                        desired_signal = -SIZE
                elif st_direction < 0 and volume_confirmed:
                    # Trend continuation with volume
                    desired_signal = -SIZE
        
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
        
        # === TIME-BASED EXIT (hold at least 4 bars = 2 days on 12h) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if supertrend flips
            if (position_side > 0 and st_direction < 0) or (position_side < 0 and st_direction > 0):
                desired_signal = 0.0
            # Exit if price crosses EMA
            if position_side > 0 and close[i] < ema_1w_aligned[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > ema_1w_aligned[i]:
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