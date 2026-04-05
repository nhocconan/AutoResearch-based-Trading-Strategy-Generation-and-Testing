#!/usr/bin/env python3
"""
exp_7099_6h_elderray_regime_v1
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with ADX regime filter.
- Bull Power = High - EMA(13); Bear Power = EMA(13) - Low
- ADX > 25 = trending market: go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
- ADX <= 25 = ranging market: fade extremes (long when Bear Power < -std, short when Bull Power > std)
- Uses 12h EMA200 as higher timeframe filter: only long when price > EMA200, short when price < EMA200
- Volume confirmation: require volume > 1.5x 20-period MA for all entries
- Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years)
- Works in both bull and bear markets via regime adaptation and HTF trend filter
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7099_6h_elderray_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EAR_LENGTH = 13          # Elder Ray EMA period
ADX_PERIOD = 14          # ADX calculation period
ADX_TREND_THRESHOLD = 25 # ADX > 25 = trending
EMA200_PERIOD = 200      # 12h EMA200 for trend filter
VOL_MA_PERIOD = 20       # Volume MA for confirmation
VOL_THRESHOLD = 1.5      # Volume must be > 1.5x MA
SIGNAL_SIZE = 0.25       # Position size (25% of capital)
ATR_PERIOD = 14          # ATR for stoploss
ATR_STOP_MULTIPLIER = 2.5 # Stoploss distance
MAX_HOLD_BARS = 50       # ~50 * 6h = 12.5 days max hold

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA200
    close_12h = df_12h['close'].values
    ema_12h_200 = pd.Series(close_12h).ewm(span=EMA200_PERIOD, adjust=False, min_periods=EMA200_PERIOD).mean().values
    
    # Align 12h EMA200 to 6h
    ema_12h_200_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray: EMA13 of close
    ema13 = pd.Series(close).ewm(span=EAR_LENGTH, adjust=False, min_periods=EAR_LENGTH).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # Smoothed Bull/Bear Power (EMA of the powers)
    bull_power_smooth = pd.Series(bull_power).ewm(span=EAR_LENGTH, adjust=False, min_periods=EAR_LENGTH).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=EAR_LENGTH, adjust=False, min_periods=EAR_LENGTH).mean().values
    
    # ADX calculation
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smoothed TR, DM+
    tr_period = pd.Series(tr).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # ATR for stoploss
    atr = tr_period  # Already calculated above
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(EAR_LENGTH, ADX_PERIOD, EMA200_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_12h_200_aligned[i]):
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
        vol_confirmed = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Skip if no volume confirmation
        if not vol_confirmed:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Determine market regime
        is_trending = adx[i] > ADX_TREND_THRESHOLD
        
        # Determine HTF trend bias from 12h EMA200
        price_above_ema200 = close[i] > ema_12h_200_aligned[i]
        price_below_ema200 = close[i] < ema_12h_200_aligned[i]
        
        # Calculate power statistics for ranging regime
        # Use rolling std of smoothed powers for dynamic thresholds
        if i >= 50:  # Need sufficient history for stats
            bull_std = pd.Series(bull_power_smooth[max(0, i-49):i+1]).std()
            bear_std = pd.Series(bear_power_smooth[max(0, i-49):i+1]).std()
        else:
            bull_std = 0.0
            bear_std = 0.0
        
        # Initialize signal
        new_position = position
        
        if position == 0:  # Looking to enter
            if is_trending:
                # Trending market: follow the stronger power
                if bull_power_smooth[i] > 0 and bull_power_smooth[i] > bear_power_smooth[i]:
                    # Bull power positive and stronger than bear power
                    if price_above_ema200:  # Only long if above HTF EMA200
                        new_position = 1
                elif bear_power_smooth[i] > 0 and bear_power_smooth[i] > bull_power_smooth[i]:
                    # Bear power positive and stronger than bull power
                    if price_below_ema200:  # Only short if below HTF EMA200
                        new_position = -1
            else:
                # Ranging market: fade extremes
                # Long when bear power is extremely negative (oversold)
                if bear_power_smooth[i] < -bull_std and price_above_ema200 * 0.5 + price_below_ema200 * 0.5 > 0:
                    # Allow both directions in ranging but prefer mean reversion to mean
                    if bear_power_smooth[i] < -bull_std:
                        new_position = 1
                # Short when bull power is extremely positive (overbought)
                elif bull_power_smooth[i] > bear_std:
                    new_position = -1
        else:
            # Managing existing position
            if position == 1:  # Long position
                # Exit if bull power fades or turns negative
                if bull_power_smooth[i] <= 0 or bear_power_smooth[i] > bull_power_smooth[i]:
                    new_position = 0
            elif position == -1:  # Short position
                # Exit if bear power fades or turns negative
                if bear_power_smooth[i] <= 0 or bull_power_smooth[i] > bear_power_smooth[i]:
                    new_position = 0
        
        # Apply position change
        if new_position != position:
            position = new_position
            if position == 0:
                signals[i] = 0.0
                entry_price = 0.0
                bars_since_entry = 0
            else:
                signals[i] = position * SIGNAL_SIZE
                entry_price = close[i]
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals