#!/usr/bin/env python3
"""
Experiment #014: 1h RSI Extremes + 4h Donchian Trend + Volume Spike

HYPOTHESIS: 1h RSI extremes (oversold/overbought) combined with 4h trend direction
and volume confirmation capture institutional reversals in trending markets.
- Long: 4h Donchian bullish + 1h RSI<25 + volume spike (institutions buying dip)
- Short: 4h Donchian bearish + 1h RSI>75 + volume spike (institutions selling rally)

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Long at RSI oversold when 4h trend up (buying dips in uptrend)
- Bear: Short at RSI overbought when 4h trend down (selling rallies in downtrend)
- Volume spike confirms institutional participation, not retail noise
- Session filter (08-20 UTC) reduces noise from low-liquidity hours

TARGET: 75-200 total trades over 4 years (~19-50/year)
Key: Tight RSI thresholds (25/75) + volume spike (1.8x) + 4h trend filter

FIXES from #009 failure (2443 trades - WAY too many):
1. Much tighter RSI thresholds (25/75 vs pivot zones that trigger constantly)
2. Stricter volume (1.8x vs 1.3x)
3. Session filter to cut overnight noise
4. 4h Donchian trend is more restrictive than HMA crossover
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_extreme_4h_donchian_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """RSI calculation"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Use pandas for proper EMA with min_periods
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        elif avg_gain[i] > 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 50.0
    
    return rsi

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

def calculate_donchian_channels(high, low, period=20):
    """Donchian channel: upper = highest high, lower = lowest low"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian channels for trend direction
    donchian_upper_4h, donchian_lower_4h = calculate_donchian_channels(
        df_4h['high'].values,
        df_4h['low'].values,
        period=20
    )
    
    # Align to 1h (with shift(1) for completed bars)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Calculate 1h indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Session filter: extract hours from datetime index (already datetime64)
    hours = prices.index.hour
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4H TREND DIRECTION (Donchian) ===
        # Bullish: price above Donchian midline
        # Bearish: price below Donchian midline
        donchian_mid = (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2.0
        trend_bullish = close[i] > donchian_mid
        trend_bearish = close[i] < donchian_mid
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8  # Strict: 1.8x average
        
        # === RSI EXTREMES ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 25  # Very oversold
        rsi_overbought = rsi > 75  # Very overbought
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + volume spike
        if trend_bullish and rsi_oversold and vol_spike:
            desired_signal = SIZE
        
        # SHORT: 4h bearish + RSI overbought + volume spike
        if trend_bearish and rsi_overbought and vol_spike:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            # If same side, keep position (no signal churn)
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