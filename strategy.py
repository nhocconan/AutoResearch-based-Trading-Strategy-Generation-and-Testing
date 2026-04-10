#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1w ADX regime filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1w ADX > 25 (trending) AND 6h volume > 1.5x 20-bar avg
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1w ADX > 25 (trending) AND 6h volume > 1.5x 20-bar avg
# - Exit when Elder Ray power crosses zero (momentum shift)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Elder Ray measures bull/bear power relative to trend; 1w ADX ensures we only trade strong trends
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: trends persist across regimes, volume confirms conviction

name = "6h_1w_elder_ray_adx_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return smoothed
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # ADX > 25 indicates strong trend
    adx_strong = adx > 25
    
    # Align 1w ADX regime to 6h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_1w, adx_strong)
    
    # Pre-compute Elder Ray on 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Power momentum: rising bull power (more positive) or falling bear power (more negative)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_falling = bear_power < np.roll(bear_power, 1)
    bull_power_rising[0] = False
    bear_power_falling[0] = False
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_strong_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_rising[i]) or np.isnan(bear_power_falling[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new trend entries
            # Long when Bull Power > 0 AND rising AND strong trend AND volume spike
            if (bull_power[i] > 0 and 
                bull_power_rising[i] and 
                adx_strong_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power < 0 AND falling AND strong trend AND volume spike
            elif (bear_power[i] < 0 and 
                  bear_power_falling[i] and 
                  adx_strong_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power crosses zero
            # Exit when Bull Power crosses below 0 (for long) or Bear Power crosses above 0 (for short)
            if position == 1:
                exit_signal = bull_power[i] <= 0
            else:  # position == -1
                exit_signal = bear_power[i] >= 0
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals