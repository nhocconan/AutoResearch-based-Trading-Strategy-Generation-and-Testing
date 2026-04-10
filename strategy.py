#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h ADX regime filter and volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (strong trend) AND volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND Bull Power < 0 AND 12h ADX > 25 (strong trend) AND volume > 1.5x 20-bar avg
# - Exit when |Bull Power| < 0.5 * ATR(14) AND |Bear Power| < 0.5 * ATR(14) (momentum exhaustion)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Elder Ray captures bull/bear strength via price relative to EMA; ADX ensures we only trade strong trends
# - Volume confirmation avoids low-liquidity false signals
# - Works in both bull and bear markets: trends persist across regimes, ADX filter avoids chop

name = "6h_12h_elder_ray_adx_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h ADX(14) trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h > 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h > 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    
    # ADX > 25 indicates strong trend
    adx_strong = adx > 25
    
    # Align 12h ADX regime to 6h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_12h, adx_strong)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Pre-compute 6h Elder Ray Index: Bull Power and Bear Power
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Pre-compute 6h ATR(14) for exit condition
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Elder Ray conditions
    bull_strong = bull_power > 0
    bear_strong = bear_power > 0
    bull_weak = np.abs(bull_power) < (0.5 * atr_6h)
    bear_weak = np.abs(bear_power) < (0.5 * atr_6h)
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_strong_aligned[i]) or np.isnan(bull_strong[i]) or np.isnan(bear_strong[i]) or
            np.isnan(bull_weak[i]) or np.isnan(bear_weak[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new trend entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND 12h strong trend AND volume spike
            if (bull_strong[i] and 
                not bear_strong[i] and  # Bear Power < 0
                adx_strong_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 AND 12h strong trend AND volume spike
            elif (bear_strong[i] and 
                  not bull_strong[i] and  # Bull Power < 0
                  adx_strong_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on momentum exhaustion
            # Exit when both Bull and Bear Power show weakness (|Power| < 0.5*ATR)
            exit_signal = bull_weak[i] and bear_weak[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals