#!/usr/bin/env python3
"""
Experiment #190: 1h Primary + 4h/12h HTF — Regime-Adaptive with Session/Volume Filters

Hypothesis: Previous 1h strategies (#180, #185, #188) failed with 0 trades due to
OVERLY STRICT entry conditions. This strategy loosens thresholds while maintaining
multi-timeframe confluence:

1. 4h HMA(21) = TREND DIRECTION (only trade with HTF trend)
2. 12h Choppiness = REGIME (range vs trend logic)
3. 1h RSI(7) = ENTRY TIMING (looser thresholds: 30/70 not 10/90)
4. Volume > 0.8x avg = CONFIRMATION
5. Session 8-20 UTC = LIQUIDITY FILTER

Key changes from failed 1h attempts:
- RSI thresholds: 30/70 (not 20/80 or 10/90)
- CHOP thresholds: 50/60 (not 38/61.8)
- Hold logic: maintain position while HTF trend intact
- Position sizing: 0.20/0.30 discrete levels

TARGET: 40-70 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_session_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Using looser thresholds: >55 range, <45 trend
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_ma + 1e-10)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    return vol_ratio

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    open_time_ms = prices["open_time"].values
    # Convert to hours since epoch, then mod 24
    hours_since_epoch = (open_time_ms / (1000 * 3600)).astype(int)
    utc_hour = hours_since_epoch % 24
    return utc_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 1h
    vol_ratio = calculate_volume_ratio(volume, period=20)
    utc_hour = get_hour_from_open_time(prices)
    
    # Calculate 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 12h Choppiness for regime
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_7[i]):
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(chop_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during liquid hours
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === VOLUME FILTER ===
        # Volume must be at least 0.8x average
        volume_ok = vol_ratio[i] >= 0.8
        
        # === HTF TREND DIRECTION (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (12h CHOP) ===
        is_range = chop_12h_aligned[i] > 55.0  # Ranging market
        is_trend = chop_12h_aligned[i] < 45.0  # Trending market
        # Neutral regime: 45-55 (use trend-following logic)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only consider entries during session with volume confirmation
        if in_session and volume_ok:
            if is_range:
                # MEAN REVERSION MODE
                # Long: RSI < 35 + price above 4h HMA (bullish bias)
                if rsi_7[i] < 35.0 and price_above_hma_4h:
                    new_signal = POSITION_SIZE_HALF
                
                # Short: RSI > 65 + price below 4h HMA (bearish bias)
                elif rsi_7[i] > 65.0 and price_below_hma_4h:
                    new_signal = -POSITION_SIZE_HALF
            
            else:
                # TREND FOLLOWING MODE (including neutral regime)
                # Long: RSI < 45 (pullback) + price above 4h HMA
                if rsi_7[i] < 45.0 and price_above_hma_4h:
                    new_signal = POSITION_SIZE_FULL
                
                # Short: RSI > 55 (pullback) + price below 4h HMA
                elif rsi_7[i] > 55.0 and price_below_hma_4h:
                    new_signal = -POSITION_SIZE_FULL
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and HTF trend still valid (relaxed entry requirements)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 4h HMA
                if price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 4h HMA
                if price_below_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 4h HMA (trend changed)
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        # Exit short if price crosses above 4h HMA (trend changed)
        if in_position and position_side < 0 and price_above_hma_4h:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals