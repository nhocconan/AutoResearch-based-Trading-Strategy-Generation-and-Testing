#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Primary signal: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
# - Regime filter: 1d ADX > 25 (trending market) enables Elder Ray signals
# - In trending markets: Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# - Works in bull/bear: ADX filter avoids whipsaws in ranging markets, captures strong trends
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - No stoploss: exit on opposite signal (mean reversion via Elder Ray)

name = "6h_1d_elderray_adx_v1"
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
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_trending = adx > 25
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_6h = prices['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13  # Higher = stronger bulls
    bear_power = low_6h - ema_13   # Lower (more negative) = stronger bears
    
    # Elder Ray signals: look for divergence from zero with momentum
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_falling = bear_power < np.roll(bear_power, 1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_trending_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power rises above -0.5 * ATR(20) approximation OR opposite signal
            atr_approx = np.abs(high_6h[i] - low_6h[i])  # simple range proxy
            if bear_power[i] > -0.5 * atr_approx or bear_power[i] > np.roll(bear_power, 1)[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power falls below 0.5 * ATR(20) approximation OR opposite signal
            atr_approx = np.abs(high_6h[i] - low_6h[i])
            if bull_power[i] < 0.5 * atr_approx or bull_power[i] < np.roll(bull_power, 1)[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals in trending market (ADX > 25)
            if adx_trending_aligned[i]:
                # Long: Bull Power positive AND rising (bulls in control and gaining strength)
                if bull_power[i] > 0 and bull_power_rising[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: Bear Power negative AND falling (bears in control and gaining strength)
                elif bear_power[i] < 0 and bear_power_falling[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals