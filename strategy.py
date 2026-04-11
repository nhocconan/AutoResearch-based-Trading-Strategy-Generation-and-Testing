#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Alligator with 1d regime filter
# - Elder Ray (Bull/Bear Power) measures trend strength via EMA13
# - Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets
# - 1d ADX > 25 confirms strong trend regime for entries
# - Long: Bull Power > 0 AND Lips > Teeth > Jaw (bullish alignment) AND 1d ADX > 25
# - Short: Bear Power < 0 AND Lips < Teeth < Jaw (bearish alignment) AND 1d ADX > 25
# - Exit: Opposite Elder Ray signal or Alligator convergence (|Lips-Jaw| < 0.1% of price)
# - Discrete position sizing: ±0.25 to control drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Works in both bull/bear markets by requiring strong trend regime (ADX filter)
# - Elder Ray captures momentum, Alligator filters false signals, ADX ensures sufficient trend strength

name = "6h_1d_elder_ray_alligator_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d ADX for regime filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr14 = wilders_smoothing(tr_1d, 14)
    dm_plus14 = wilders_smoothing(dm_plus, 14)
    dm_minus14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_14 = wilders_smoothing(dx, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Pre-compute 6h indicators
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA (Wilder's smoothing)
    def smma(data, period):
        return wilders_smoothing(data, period)
    
    jaw = smma(smma(high, 13), 8)  # Jaw: Smma(Median Price,13) then Smma(,8)
    teeth = smma(smma(low, 8), 5)   # Teeth: Smma(Median Price,8) then Smma(,5)
    lips = smma(smma((high + low) / 2, 5), 3)  # Lips: Smma(Median Price,5) then Smma(,3)
    
    # Elder Ray
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_14_aligned[i] > 25
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray signals
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull Power > 0 AND bullish Alligator alignment AND strong trend
        if bull_power_positive and bullish_alignment and strong_trend:
            enter_long = True
        
        # Short: Bear Power < 0 AND bearish Alligator alignment AND strong trend
        if bear_power_negative and bearish_alignment and strong_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Bear Power turns negative OR Alligator converges (|Lips-Jaw| < 0.1% of price)
            exit_long = (bear_power[i] >= 0) or (np.abs(lips[i] - jaw[i]) < 0.001 * close[i])
        elif position == -1:
            # Exit short if Bull Power turns positive OR Alligator converges
            exit_short = (bull_power[i] <= 0) or (np.abs(lips[i] - jaw[i]) < 0.001 * close[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals