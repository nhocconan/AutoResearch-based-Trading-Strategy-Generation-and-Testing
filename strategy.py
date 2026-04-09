#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Regime Filter
# - Uses 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for momentum
# - Filters by 1w ADX regime: ADX > 25 = trending (trade Elder Ray signals), ADX < 20 = ranging (fade Elder Ray extremes)
# - Weekly pivot (based on prior week) provides directional bias: long only above weekly pivot, short only below
# - Volume confirmation: volume > 1.5 * 20-period average
# - Designed to work in bull markets via trend-following Elder Ray signals and in bear markets via mean reversion in ranging regimes
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag on 6h timeframe
# - Elder Ray captures institutional buying/selling pressure better than simple price crosses

name = "6h_1w_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w ADX for regime detection (trending vs ranging)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr_1w, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w > 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w > 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1w ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1d Weekly pivot levels (based on prior week's OHLC)
    # Need to resample 1d to get weekly OHLC from daily data
    # Since we have daily data, we can calculate weekly pivot from last 5 days approx
    # But better: use actual weekly data from mtf_data - we already have df_1w
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # 6h Elder Ray components
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 6h ATR(14) for dynamic thresholds
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_6h[i]) or atr_6h[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit conditions
            if is_ranging and bear_power[i] < 0.5 * atr_6h[i]:  # Fade long in ranging market when bear power weakens
                position = 0
                signals[i] = 0.0
            elif close[i] < weekly_pivot_aligned[i]:  # Break below weekly pivot
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if is_ranging and bull_power[i] < 0.5 * atr_6h[i]:  # Fade short in ranging market when bull power weakens
                position = 0
                signals[i] = 0.0
            elif close[i] > weekly_pivot_aligned[i]:  # Break above weekly pivot
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries based on regime and Elder Ray
            if is_trending:
                # Trending regime: trade with Elder Ray momentum
                if bull_power[i] > atr_6h[i] and close[i] > weekly_pivot_aligned[i] and volume_confirm[i]:
                    position = 1
                    signals[i] = 0.25
                elif bear_power[i] > atr_6h[i] and close[i] < weekly_pivot_aligned[i] and volume_confirm[i]:
                    position = -1
                    signals[i] = -0.25
            else:  # ranging regime
                # Ranging regime: fade Elder Ray extremes at weekly pivot levels
                if bull_power[i] > atr_6h[i] and close[i] < weekly_pivot_aligned[i] and volume_confirm[i]:
                    # Strong bull power but below pivot = potential fade short
                    position = -1
                    signals[i] = -0.25
                elif bear_power[i] > atr_6h[i] and close[i] > weekly_pivot_aligned[i] and volume_confirm[i]:
                    # Strong bear power but above pivot = potential fade long
                    position = 1
                    signals[i] = 0.25
    
    return signals