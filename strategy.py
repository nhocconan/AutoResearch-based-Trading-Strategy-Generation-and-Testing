#!/usr/bin/env python3
"""
Experiment #026: 6h VWAP Anchored + RSI Extreme + Choppiness Regime

HYPOTHESIS: Daily VWAP anchors institutional "fair value" that price oscillates around.
During trending markets (low Choppiness), RSI extremes (<30, >70) mark exhaustion points
that often reverse. Using 1d VWAP direction as trend bias, entering on RSI extremes only
in the direction of trend catches mean-reversion moves within trends. This is different
from momentum strategies (MACD, TRIX) and channel strategies (Donchian) that failed.

Why it should work in both bull AND bear:
- Bull: Pullbacks to VWAP with RSI < 30 = oversold buying opportunity
- Bear: Rallies to VWAP with RSI > 70 = short opportunities
- Range: RSI extremes still work for bounces off extremes

TIMEFRAME: 6h primary
HTF: 1d for VWAP and Choppiness regime
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vwap_rsi_chop_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_vwap(high, low, close, volume):
    """VWAP calculated per bar - typical price weighted by volume"""
    typical_price = (high + low + close) / 3.0
    return np.sum(typical_price * volume) / np.sum(volume) if np.sum(volume) > 0 else typical_price[-1]

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP < 38.2 = trending (good for trend following)
    CHOP > 61.8 = choppy (avoid, good for mean reversion)
    Values around 50 = neutral
    """
    n = len(close)
    if n < period:
        return np.full(n, 50.0)  # neutral default
    
    chop = np.full(n, 50.0, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest - Lowest over period
        hl_range = np.max(high[i - period + 1:i + 1]) - np.min(low[i - period + 1:i + 1])
        
        if hl_range > 0:
            # CHOP = 100 * log10(atr_sum / hl_range) / log10(period)
            chop[i] = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """RSI with min_periods"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100 - (100 / (1 + rs))).values

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

def calculate_ema(close, span, min_periods=None):
    """EMA calculation"""
    if min_periods is None:
        min_periods = span
    return pd.Series(close).ewm(span=span, min_periods=min_periods, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d indicators (aligned to 6h) ===
    # 1d EMA21 for trend direction
    ema_1d_raw = calculate_ema(df_1d['close'].values, span=21, min_periods=21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # 1d Choppiness Index for regime
    chop_1d_raw = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # === Local 6h indicators ===
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Local VWAP approximation using rolling window (22 days ~ 1 month)
    typical_price = (high + low + close) / 3.0
    vol_price = typical_price * volume
    
    # Cumulative VWAP for current session
    cum_vol_price = np.zeros(n)
    cum_volume = np.zeros(n)
    cum_vol_price[0] = vol_price[0]
    cum_volume[0] = volume[0]
    
    for i in range(1, n):
        cum_vol_price[i] = cum_vol_price[i-1] + vol_price[i]
        cum_volume[i] = cum_volume[i-1] + volume[i]
    
    # Rolling VWAP (reset conceptually each day - use 88 bars ≈ 22 days of 6h)
    vwap_window = 88
    rolling_vol_price = pd.Series(cum_vol_price).rolling(window=vwap_window, min_periods=vwap_window).last().values
    rolling_vol = pd.Series(cum_volume).rolling(window=vwap_window, min_periods=vwap_window).last().values
    
    # Previous cum values for incremental VWAP
    prev_vol_price = np.zeros(n)
    prev_vol = np.zeros(n)
    prev_vol_price[vwap_window:] = cum_vol_price[:-vwap_window]
    prev_vol[vwap_window:] = cum_volume[:-vwap_window]
    
    vwap = (cum_vol_price - prev_vol_price) / np.where((cum_volume - prev_vol) > 0, (cum_volume - prev_vol), 1)
    
    # Price vs VWAP position
    vwap_distance = (close - vwap) / (atr_14 + 1e-10)
    
    # RSI MA for smoothing
    rsi_ma = pd.Series(rsi_14).rolling(window=5, min_periods=3).mean().values
    
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
    
    warmup = 100  # Need enough data for VWAP window
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (1d Choppiness) ===
        chop = chop_1d_aligned[i] if not np.isnan(chop_1d_aligned[i]) else 50.0
        is_trending = chop < 50.0  # Not too choppy
        is_choppy = chop > 55.0    # Very choppy - be more selective
        
        # === TREND BIAS (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        trend_bullish = price_above_1d_ema
        trend_bearish = not price_above_1d_ema
        
        # === RSI LEVELS ===
        rsi_val = rsi_14[i]
        rsi_smooth = rsi_ma[i] if not np.isnan(rsi_ma[i]) else rsi_val
        rsi_oversold = rsi_val < 35  # Expanded from 30
        rsi_overbought = rsi_val > 65  # Expanded from 70
        
        # === VWAP POSITION ===
        dist_from_vwap = vwap_distance[i]
        
        # Extreme: price significantly away from VWAP
        far_from_vwap_long = dist_from_vwap < -1.5   # Below VWAP by 1.5 ATR
        far_from_vwap_short = dist_from_vwap > 1.5   # Above VWAP by 1.5 ATR
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # RSI oversold + price below VWAP (cheap) + trend aligned OR not bearish
            if rsi_oversold:
                # Prefer: in uptrend OR not in strong downtrend
                if trend_bullish or not far_from_vwap_short:
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # RSI overbought + price above VWAP (expensive) + trend aligned OR not bullish
            if rsi_overbought:
                # Prefer: in downtrend OR not in strong uptrend
                if trend_bearish or not far_from_vwap_long:
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
        
        # === EXIT: RSI mean reversion (return to 45-55) ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: RSI normalizes (no longer oversold)
            if rsi_val > 52:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: RSI normalizes (no longer overbought)
            if rsi_val < 48:
                exit_triggered = True
        
        if exit_triggered:
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