#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Volume Regime Filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 6h data)
# - 1d Volume Regime: High volume (>1.5x 20-day average) confirms institutional participation
# - Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum) AND 1d volume confirmation
# - Short when Bear Power > 0 AND Bull Power < previous Bull Power (bearish momentum) AND 1d volume confirmation
# - Exit when power signals weaken (Bull Power < 0 for longs, Bear Power < 0 for shorts)
# - Position size: 0.25 (25% of capital) to manage drawdown in volatile 6h markets
# - Target: 12-30 trades/year on 6h (50-120 total over 4 years) to minimize fee drag
# - Works in bull/bear: Volume regime filters low-quality signals, Elder Ray adapts to momentum shifts

name = "6h_1d_elderray_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume average (20-period) for regime filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute 6h Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # EMA13 for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_6h - ema13_6h
    # Bear Power = EMA13 - Low
    bear_power = ema13_6h - low_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(volume_1d[i // 24] if i // 24 < len(volume_1d) else np.nan)):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get 1d volume for current 6h bar (aligned to completed 1d bars)
        idx_1d = i // 24  # 24x 6h bars in 1d (approximate, alignment handles precision)
        if idx_1d >= len(volume_1d):
            vol_1d_current = volume_1d[-1] if len(volume_1d) > 0 else 0
            vol_ma_1d_current = vol_ma_20_1d[-1] if len(vol_ma_20_1d) > 0 else 0
        else:
            vol_1d_current = volume_1d[idx_1d]
            vol_ma_1d_current = vol_ma_20_1d[idx_1d]
        
        # 1d volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirm = vol_1d_current > 1.5 * vol_ma_1d_current and vol_ma_1d_current > 0
        
        # Previous power values for momentum check
        prev_bull = bull_power[i-1] if i > 0 else 0
        prev_bear = bear_power[i-1] if i > 0 else 0
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bull Power increasing (momentum) AND volume confirmation
            if bull_power[i] > 0 and bull_power[i] > prev_bull and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND Bear Power increasing (momentum) AND volume confirmation
            elif bear_power[i] > 0 and bear_power[i] > prev_bear and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when Bull Power weakens (< 0)
            exit_long = bull_power[i] < 0
            # Exit short when Bear Power weakens (< 0)
            exit_short = bear_power[i] < 0
            
            if (position == 1 and exit_long) or (position == -1 and exit_short):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals