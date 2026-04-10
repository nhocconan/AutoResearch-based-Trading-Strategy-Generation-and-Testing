#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# - Primary signal: 6h Elder Ray (Bull Power > 0 and Bear Power < 0) indicates strong momentum
# - Regime filter: 12h ADX > 25 ensures we only trade in trending markets (avoids chop)
# - Volume confirmation: 6h volume > 1.5x 20-period average to ensure institutional participation
# - Position size: 0.25 discrete level to balance return and drawdown
# - Stoploss: 2.5x ATR(20) on 6h for volatility-adjusted risk control
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in both bull and bear markets by requiring strong directional momentum (Elder Ray) 
#   and trending conditions (ADX), avoiding false signals in ranging markets

name = "6h_12h_elderray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h ADX(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_filter = adx > 25  # Trending market regime
    adx_filter_aligned = align_htf_to_ltf(prices, df_12h, adx_filter)
    
    # Pre-compute 6h indicators
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema_13
    bear_power = low_6h - ema_13
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_6h > (1.5 * avg_volume_20)
    
    # ATR for stoploss
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_20 = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_filter_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Elder Ray weakening OR stoploss hit
            if bull_power[i] <= 0 or close_6h[i] < entry_price - 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray weakening OR stoploss hit
            if bear_power[i] >= 0 or close_6h[i] > entry_price + 2.5 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for strong Elder Ray signals with volume and trend filters
            if adx_filter_aligned[i] and volume_filter[i]:
                # Long: strong bullish momentum (Bull Power > 0 and Bear Power < 0)
                if bull_power[i] > 0 and bear_power[i] < 0:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: strong bearish momentum (Bear Power < 0 and Bull Power > 0 is impossible, 
                # so we use Bear Power < 0 and Bull Power < 0 for confirmation)
                elif bear_power[i] < 0 and bull_power[i] < 0:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals