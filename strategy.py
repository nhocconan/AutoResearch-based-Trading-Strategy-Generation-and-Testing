#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power with 12h regime filter and volume confirmation
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 AND volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND Bull Power < 0 AND 12h ADX > 25 AND volume > 1.5x 20-bar avg
# - Exit when power values converge (|Bull Power| < 0.1 * ATR AND |Bear Power| < 0.1 * ATR)
# - Uses 12h ADX for regime filter to avoid ranging markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Elder Ray identifies bull/bear strength via power; ADX filters for trending regimes

name = "6h_12h_elder_ray_power_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h ADX for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    period = 14
    atr_12h = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = wilders_smoothing(dx, period)
    
    # Align HTF ADX to LTF
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute ATR for exit condition (using 6h data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range for 6h
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    
    atr_6h = wilders_smoothing(tr_6h, 14)
    
    # Pre-compute EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND 12h trending AND volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                adx_12h_aligned[i] > 25 and vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND Bull Power < 0 AND 12h trending AND volume spike
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  adx_12h_aligned[i] > 25 and vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power converges
            # Exit when power values converge (weak trend)
            exit_signal = False
            if position == 1:  # Long position
                if (bull_power[i] < 0.1 * atr_6h[i] and bear_power[i] < 0.1 * atr_6h[i]):
                    exit_signal = True
            elif position == -1:  # Short position
                if (bull_power[i] < 0.1 * atr_6h[i] and bear_power[i] < 0.1 * atr_6h[i]):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals