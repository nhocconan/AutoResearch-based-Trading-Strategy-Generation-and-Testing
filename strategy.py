#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal strategy with 12h trend filter and volume confirmation.
# Camarilla pivots identify key intraday support/resistance levels.
# In ranging markets, price tends to revert from R3/S3 levels.
# In trending markets, breakouts through R4/S4 often continue.
# Use 12h EMA(50) slope to determine regime: fade at R3/S3 in range, breakout at R4/S4 in trend.
# Volume confirms institutional participation at key levels.
# Works in both bull/bear markets by adapting to regime.

name = "exp_13599_6h_camarilla_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
RANGE_ADX_THRESHOLD = 20  # ADX < 20 = range, ADX > 25 = trend

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    
    return r1, r2, r3, r4, s1, s2, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period):
    """Calculate ADX"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Handle division by zero
    adx = np.where((di_plus + di_minus) == 0, 0, adx)
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter and Camarilla levels ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA for trend direction
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # 12h ADX for regime detection
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(CAMARILLA_LOOKBACK, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_slope_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Regime detection from 12h ADX
        is_range = adx_12h_aligned[i] < RANGE_ADX_THRESHOLD
        is_trend = adx_12h_aligned[i] > (RANGE_ADX_THRESHOLD + 5)
        
        # Calculate Camarilla levels for current 6h bar (using previous bar's data to avoid look-ahead)
        if i >= CAMARILLA_LOOKBACK:
            # Use previous bar's high/low/close for current levels
            phigh = np.max(high[i-CAMARILLA_LOOKBACK:i])
            plow = np.min(low[i-CAMARILLA_LOOKBACK:i])
            pclose = close[i-1]
            
            r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(phigh, plow, pclose)
            
            # Price levels
            price = close[i]
            
            # Trading logic based on regime
            long_signal = False
            short_signal = False
            
            if is_range and volume_ok:
                # In range: fade at R3/S3
                if price <= s3 and price > s4:  # Near S3, above S4
                    long_signal = True
                elif price >= r3 and price < r4:  # Near R3, below R4
                    short_signal = True
            elif is_trend and volume_ok:
                # In trend: breakout continuation at R4/S4
                if price > r4:  # Break above R4
                    long_signal = True
                elif price < s4:  # Break below S4
                    short_signal = True
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions
            exit_signal = False
            if i >= CAMARILLA_LOOKBACK:
                phigh = np.max(high[i-CAMARILLA_LOOKBACK:i])
                plow = np.min(low[i-CAMARILLA_LOOKBACK:i])
                pclose = close[i-1]
                _, _, _, r4, _, _, s3, _ = calculate_camarilla(phigh, plow, pclose)
                
                # Exit long if price reaches S3 (profit target) or breaks below S2 (reversal)
                if close[i] <= s3 or close[i] < s2:
                    exit_signal = True
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit conditions
            exit_signal = False
            if i >= CAMARILLA_LOOKBACK:
                phigh = np.max(high[i-CAMARILLA_LOOKBACK:i])
                plow = np.min(low[i-CAMARILLA_LOOKBACK:i])
                pclose = close[i-1]
                _, _, s4, _, r3, _, _, _ = calculate_camarilla(phigh, plow, pclose)
                
                # Exit short if price reaches R3 (profit target) or breaks above R2 (reversal)
                if close[i] >= r3 or close[i] > r2:
                    exit_signal = True
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals