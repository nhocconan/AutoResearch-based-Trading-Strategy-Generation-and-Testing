#!/usr/bin/env python3
"""
Experiment #005: 12h ATR Ratio Volatility Expansion + Donchian Breakout

HYPOTHESIS: Explosive moves follow low-volatility consolidation.
ATR_ratio = ATR(5)/ATR(30) > 1.3 identifies squeeze-like conditions about to release.
Combined with 12h Donchian(20) breakout + volume confirmation + 1d trend.

WHY 12h: 3x slower than 4h = fewer trades = less fee drag.
12h captures multi-day swing trades, not noise.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Long: price breaks above Donchian upper + ATR expansion + volume spike + above 1d EMA50
- Short: price breaks below Donchian lower + ATR expansion + volume spike + below 1d EMA50
Symmetrical logic works in trending markets and reversals.

TARGET: 75-150 total trades over 4 years = 19-37/year.
Signal size: 0.25 (discrete).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_atr_ratio_donchian_vol_1d_v1"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator for reversal timing"""
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        period_high = np.max(high[i - period + 1:i + 1])
        period_low = np.min(low[i - period + 1:i + 1])
        if period_high != period_low:
            result[i] = -100 * (period_high - close[i]) / (period_high - period_low)
    
    return result

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR ratio: short-term / long-term volatility
    # ATR(5) / ATR(30) > threshold = volatility expansion after consolidation
    atr_5 = calculate_atr(high, low, close, period=5)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = atr_5 / np.where(atr_30 > 0, atr_30, 1)
    
    # Donchian channels (20 periods = 10 days on 12h)
    donchian_period = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Williams %R for reversal timing
    willr = calculate_williams_r(high, low, close, period=14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(100, donchian_period + 30)  # Need enough for Donchian + ATR
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FILTER CONDITIONS ===
        # ATR ratio > 1.3 = volatility expansion after consolidation
        atr_expansion = atr_ratio[i] > 1.3
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Trend direction (1d EMA50)
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # Donchian breakout state
        above_donchian_upper = close[i] > donchian_upper[i]
        below_donchian_lower = close[i] < donchian_lower[i]
        
        # Williams %R reversal zones
        willr_oversold = willr[i] < -80
        willr_overbought = willr[i] > -20
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Break above Donchian upper + ATR expansion + volume + trend + oversold bounce
            # Conditions: above_donchian_upper AND atr_expansion AND vol_spike AND price_above_ema AND willr_oversold
            if above_donchian_upper and atr_expansion and vol_spike and price_above_1d_ema and willr_oversold:
                desired_signal = SIZE
            # Alternative: Price bounced from lower band with same filters
            elif willr_oversold and vol_spike and atr_expansion and price_above_1d_ema:
                desired_signal = SIZE
            
            # SHORT: Break below Donchian lower + ATR expansion + volume + trend + overbought
            if below_donchian_lower and atr_expansion and vol_spike and price_below_1d_ema and willr_overbought:
                desired_signal = -SIZE
            # Alternative: Price rejected from upper band with same filters
            elif willr_overbought and vol_spike and atr_expansion and price_below_1d_ema:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (optional: trail stop after 2R profit) ===
        bars_held = i - entry_bar
        
        # Minimum 2 bars to avoid fee churn
        if in_position and bars_held >= 2:
            # Exit if Williams %R reaches opposite extreme
            if position_side > 0 and willr[i] > -20:
                desired_signal = 0.0
            if position_side < 0 and willr[i] < -80:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = close[i] - 2.0 * entry_atr
                else:
                    stop_price = close[i] + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals