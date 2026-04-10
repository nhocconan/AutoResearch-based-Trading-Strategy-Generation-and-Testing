#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX(14) > 25 (strong trend) AND 6h volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX(14) > 25 AND 6h volume > 1.5x 20-bar avg
# - Exit when either power crosses zero (trend weakening) OR volume drops below average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Elder Ray measures bull/bear strength relative to EMA; ADX filters for trending markets only
# - Volume confirmation ensures institutional participation
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in bull markets (strong uptrends) and bear markets (strong downtrends), avoids ranging markets

name = "6h_1d_elder_ray_adx_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        alpha = 1.0 / period
        for i in range(len(values)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = values[i]
            else:
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    
    # ADX > 25 indicates strong trend
    adx_strong = adx > 25
    
    # Align 1d ADX regime to 6h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    # Pre-compute Elder Ray on 6h data: Bull Power and Bear Power
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = EMA(13) - Low
    bear_power = ema_13 - low
    
    # Elder Ray signals
    bull_power_pos = bull_power > 0
    bear_power_pos = bear_power > 0
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_strong_aligned[i]) or np.isnan(bull_power_pos[i]) or
            np.isnan(bear_power_pos[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new trend entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX strong trend AND volume spike
            if (bull_power_pos[i] and 
                not bear_power_pos[i] and  # Bear Power < 0
                adx_strong_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX strong trend AND volume spike
            elif (bear_power_pos[i] and 
                  not bull_power_pos[i] and  # Bull Power < 0
                  adx_strong_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when either power crosses zero (trend weakening) OR volume drops below average
            exit_signal = (not bull_power_pos[i] and position == 1) or \
                          (not bear_power_pos[i] and position == -1) or \
                          (not vol_spike[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals