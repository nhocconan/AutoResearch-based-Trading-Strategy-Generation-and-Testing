#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with ATR-based position sizing and volume confirmation.
# Weekly Donchian (20-period) captures major trend structure that works in both bull and bear markets.
# Breakouts above weekly high or below weekly low with volume confirmation (>1.5x 20-day average volume) signal strong momentum.
# ATR filter ensures sufficient volatility to avoid choppy low-volume false breakouts.
# Position size scales inversely with ATR to maintain consistent risk (target ~0.25 at median ATR).
# Stop loss placed at opposite weekly Donchian band to limit risk per trade.
# Target: 20-50 trades over 4 years (5-12/year) on 1d timeframe to minimize fee drag.

name = "1d_1w_donchian_atr_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian Channel (20-period)
    # Upper band = highest high over past 20 weekly candles
    # Lower band = lowest low over past 20 weekly candles
    high_ma_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w ATR (14-period) for volatility filtering and position sizing
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1w indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, high_ma_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, low_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Pre-compute daily volume confirmation (20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i]) or
            atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 20-day volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 50-day average (avoid low-vol chop)
        if i >= 50:
            atr_ma_50 = pd.Series(atr_aligned[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1]
            vol_filter = atr_aligned[i] > atr_ma_50
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Dynamic position size: inverse volatility scaling (target ~0.25 at median ATR)
        # Clamp ATR to reasonable range to avoid extreme position sizes
        atr_clamped = np.clip(atr_aligned[i], 0.001, 0.50)  # Avoid division by zero or tiny ATR
        base_size = 0.25
        vol_scaling = 0.01 / atr_clamped  # Scale so 1% ATR gives ~0.25 size
        vol_scaling = np.clip(vol_scaling, 0.5, 2.0)  # Clamp scaling to reasonable range
        position_size = base_size * vol_scaling
        position_size = np.clip(position_size, 0.15, 0.35)  # Final clamp to 0.15-0.35
        
        if position == 1:  # Long position
            # Exit on retracement to weekly lower band or stop at opposite band
            if close[i] < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to weekly upper band or stop at opposite band
            if close[i] > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Weekly Donchian breakout with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above weekly upper band -> long
                if close[i] > upper_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below weekly lower band -> short
                elif close[i] < lower_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals