#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Donchian channel breakout with weekly EMA filter and volume confirmation
# Weekly Donchian(20) identifies major structural breaks that work in both bull and bear markets
# Weekly EMA50 filter ensures we only trade in the direction of the weekly trend
# Volume confirmation (current 6h volume > 1.5x 20-period average) filters false breakouts
# ATR-based position sizing adjusts for volatility to maintain consistent risk
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1w_donchian_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channel (20-period)
    # Upper channel = highest high over 20 periods
    # Lower channel = lowest low over 20 periods
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1w ATR(14) for volatility filtering and position sizing
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian channels, EMA, and ATR to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, high_roll)
    lower_aligned = align_htf_to_ltf(prices, df_1w, low_roll)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma_20[i]) or atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 30-period average (avoid low-vol chop)
        atr_ma_30 = pd.Series(atr_aligned).rolling(window=30, min_periods=30).mean()
        if len(atr_ma_30) > i:
            vol_filter = atr_aligned[i] > atr_ma_30.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Dynamic position size: inverse volatility scaling (target ~0.25 at median ATR)
        # Clamp ATR to reasonable range to avoid extreme position sizes
        atr_clamped = np.clip(atr_aligned[i], 0.001, 0.10)  # Avoid division by zero or tiny ATR
        base_size = 0.25
        vol_scaling = 0.01 / atr_clamped  # Scale so 1% ATR gives ~0.25 size
        vol_scaling = np.clip(vol_scaling, 0.5, 2.0)  # Clamp scaling to reasonable range
        position_size = base_size * vol_scaling
        position_size = np.clip(position_size, 0.15, 0.35)  # Final clamp to 0.15-0.35
        
        if position == 1:  # Long position
            # Exit on retracement to lower Donchian channel or weekly EMA
            if close[i] < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < ema_aligned[i]:  # Stop loss below weekly EMA
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to upper Donchian channel or weekly EMA
            if close[i] > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > ema_aligned[i]:  # Stop loss above weekly EMA
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with EMA filter and volume confirmation
            # Only trade in direction of weekly EMA trend
            if volume_confirmed:
                # Long breakout: price breaks above upper Donchian and above weekly EMA
                if close[i] > upper_aligned[i] and close[i] > ema_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short breakout: price breaks below lower Donchian and below weekly EMA
                elif close[i] < lower_aligned[i] and close[i] < ema_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals