#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 12h ADX trend filter and volume confirmation.
- Uses Ichimoku Cloud (Tenkan/Kijun/Senkou Span A/B) from 6h for entry timing.
- Breakout above/below Cloud with price >/< Kumo twist signals strong momentum.
- Trend filter: 12h ADX > 25 ensures we only trade in trending markets.
- Volume confirmation: > 2.0x 20-bar average filters weak breakouts.
- Designed for 6h timeframe to capture medium-term trends with controlled frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-30 trades/year (50-120 total over 4 years) to stay fee-efficient.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 6h timeframe (wait for 12h bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Ichimoku Cloud components (6h)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Current Kumo (Cloud) boundaries: Senkou Span A/B shifted 26 periods ahead
    # For signal at index i, we use Senkou A/B from index i-26 (already published)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Kumo twist: Senkou A > Senkou B (bullish cloud) or Senkou A < Senkou B (bearish cloud)
    kumo_twist_bullish = senkou_a_shifted > senkou_b_shifted
    kumo_twist_bearish = senkou_a_shifted < senkou_b_shifted
    
    # Price relative to Kumo
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 52 + 26)  # Senkou B needs 52 + 26 shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX trend filter (> 25 = trending market)
        trending = adx_aligned[i] > 25
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price above Kumo AND bullish twist AND Tenkan > Kijun AND trending AND volume
            if (price_above_kumo[i] and kumo_twist_bullish[i] and tenkan[i] > kijun[i] and 
                trending and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price below Kumo AND bearish twist AND Tenkan < Kijun AND trending AND volume
            elif (price_below_kumo[i] and kumo_twist_bearish[i] and tenkan[i] < kijun[i] and 
                  trending and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below Kumo OR Tenkan < Kijun
            if price_below_kumo[i] or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above Kumo OR Tenkan > Kijun
            if price_above_kumo[i] or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0