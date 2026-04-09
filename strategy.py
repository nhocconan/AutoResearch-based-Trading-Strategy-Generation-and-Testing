#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Elder Ray (Bull/Bear Power) from 6h: measures buying/selling pressure vs EMA13
# - ADX from 1d: trend strength filter (ADX > 25 = trending market)
# - Entry: Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25
# - Exit: Reverse signal or ADX < 20 (trend weakening)
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Novelty: Combines Elder Ray momentum with 1d ADX regime to avoid whipsaws in ranging markets
# - Works in both bull/bear: Elder Ray captures momentum shifts, ADX filter ensures trades only in trending conditions

name = "6h_1d_elderray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for trend regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Rest: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1d = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d > 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d > 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = WilderSmoothing(dx, 14)
    
    # Trend regime: ADX > 25 = trending market
    trending_regime = adx_1d > 25
    
    # Align 1d trend regime to 6h timeframe (completed 1d bar only)
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure (negative values indicate selling)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(trending_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: reverse signal OR trend weakening (ADX < 20)
            if (bull_power[i] <= 0 and bear_power[i] >= 0) or trending_regime_aligned[i] == False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: reverse signal OR trend weakening (ADX < 20)
            if (bull_power[i] >= 0 and bear_power[i] <= 0) or trending_regime_aligned[i] == False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with trend regime filter
            # Long: Bull Power > 0 (buying pressure) AND Bear Power < 0 (weak selling) AND trending market
            if bull_power[i] > 0 and bear_power[i] < 0 and trending_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: Bull Power < 0 (weak buying) AND Bear Power > 0 (selling pressure) AND trending market
            elif bull_power[i] < 0 and bear_power[i] > 0 and trending_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals