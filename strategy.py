#!/usr/bin/env python3
"""
exp_6759_6h_elder_ray_regime_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) combined with 12h ADX regime filter.
In trending markets (ADX > 25): trade in direction of Elder Ray (long when Bull Power > 0, short when Bear Power < 0).
In ranging markets (ADX <= 25): fade extreme Elder Ray readings (long when Bear Power < -std, short when Bull Power > +std).
Volume confirmation required for all entries. Designed for 6h timeframe to capture medium-term swings with ~12-37 trades/year (50-150 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn. ATR-based stoploss manages risk.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6759_6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 50  # ~12.5 days (6h bars)
STD_DEV_PERIOD = 20
STD_DEV_MULTIPLIER = 1.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for ADX regime
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX for regime detection
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    
    # Smooth DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    tr_smooth = pd.Series(tr_12h).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align ADX to LTF (6h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA
    ema_close = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bull_power = high - ema_close
    bear_power = low - ema_close
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Standard deviation of Bear Power for ranging regime thresholds
    bear_power_series = pd.Series(bear_power)
    bear_power_std = bear_power_series.rolling(window=STD_DEV_PERIOD, min_periods=STD_DEV_PERIOD).std().values
    
    # Standard deviation of Bull Power for ranging regime thresholds
    bull_power_series = pd.Series(bull_power)
    bull_power_std = bull_power_series.rolling(window=STD_DEV_PERIOD, min_periods=STD_DEV_PERIOD).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, ADX_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, STD_DEV_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Regime detection: trending vs ranging
        is_trending = adx_aligned[i] > ADX_TREND_THRESHOLD
        is_ranging = adx_aligned[i] <= ADX_TREND_THRESHOLD
        
        # Initialize signal
        signals[i] = 0.0
        
        if is_trending:
            # Trending regime: trade with Elder Ray direction
            long_signal = bull_power[i] > 0 and vol_confirmed
            short_signal = bear_power[i] < 0 and vol_confirmed
            
            if long_signal and position <= 0:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal and position >= 0:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif position == 1:
                signals[i] = SIGNAL_SIZE
            elif position == -1:
                signals[i] = -SIGNAL_SIZE
                
        else:  # ranging regime
            # Fade extreme Elder Ray readings
            long_signal = bear_power[i] < -STD_DEV_MULTIPLIER * bear_power_std[i] and vol_confirmed
            short_signal = bull_power[i] > STD_DEV_MULTIPLIER * bull_power_std[i] and vol_confirmed
            
            if long_signal and position <= 0:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal and position >= 0:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif position == 1:
                signals[i] = SIGNAL_SIZE
            elif position == -1:
                signals[i] = -SIGNAL_SIZE
    
    return signals