#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX trend filter + volume confirmation
# - Primary signal: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
#   Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# - Trend filter: 1d ADX > 25 ensures we only trade in trending markets (avoid chop)
# - Volume confirmation: 6h volume > 20-period median volume (avoid low-participation signals)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Elder Ray captures momentum shifts, ADX filter ensures
#   trades align with strong trends, reducing false signals in ranging markets

name = "6h_1d_elderray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA14 for ADX calculation
    ema_14_1d = pd.Series(close_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d True Range and ATR14 for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d +DI and -DI for ADX
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14_1d
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14_1d
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe (completed 1d bar only)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # 6h Elder Ray slope (1-period change)
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    bull_power_slope[0] = bear_power_slope[0] = 0
    
    # 6h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_14_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(bull_power_slope[i]) or
            np.isnan(bear_power_slope[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Elder Ray turns bearish OR ADX weakens (<25) OR volume dries up
            if (bull_power[i] <= 0 or bull_power_slope[i] <= 0 or
                adx_14_aligned[i] < 25.0 or not volume_regime[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray turns bullish OR ADX weakens (<25) OR volume dries up
            if (bear_power[i] >= 0 or bear_power_slope[i] >= 0 or
                adx_14_aligned[i] < 25.0 or not volume_regime[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray extremes with volume confirmation and ADX filter
            # Long: Bull Power > 0 AND rising AND ADX > 25 AND volume regime
            if (bull_power[i] > 0.0 and bull_power_slope[i] > 0.0 and
                adx_14_aligned[i] > 25.0 and volume_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power < 0 AND falling AND ADX > 25 AND volume regime
            elif (bear_power[i] < 0.0 and bear_power_slope[i] < 0.0 and
                  adx_14_aligned[i] > 25.0 and volume_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals