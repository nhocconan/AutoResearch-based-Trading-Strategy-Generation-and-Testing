#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channels with ATR-based position sizing and volume confirmation
# Donchian(20) from 1w provides major support/resistance levels that work in both bull and bear markets
# ATR filter ensures we only trade when volatility is sufficient (avoid choppy low-vol periods)
# Volume confirmation (current 1d volume > 1.5x 20-period average) filters false breakouts
# Position size scales with volatility (inverse ATR) to maintain consistent risk
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)

name = "1d_1w_donchian_atr_volume_v2"
timeframe = "1d"
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
    if len(df_1w) < 25:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1w ATR (14-period) for volatility filtering and position sizing
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian levels and ATR to 1d timeframe
    dh_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    dm_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or
            np.isnan(dm_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(vol_ma_20[i]) or atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low-vol chop)
        atr_ma_50 = pd.Series(atr_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50) > i:
            vol_filter = atr_aligned[i] > atr_ma_50.iloc[i]
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
            # Exit on retracement to Donchian midpoint or lower band
            if close[i] < dm_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to Donchian midpoint or upper band
            if close[i] > dm_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout trading with volume and volatility confirmation
            # Long on Donchian high breakout, Short on Donchian low breakout
            if volume_confirmed:
                if close[i] > dh_aligned[i]:
                    position = 1
                    signals[i] = position_size
                elif close[i] < dl_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals