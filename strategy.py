#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter
# - Uses 1d HTF for regime: ADX>25 = trending, ADX<20 = ranging
# - 6h Elder Ray: Bull Power = Close - EMA13, Bear Power = EMA13 - Close
# - Long when Bull Power > 0 and Bear Power < 0 in trending up (ADX rising)
# - Short when Bear Power > 0 and Bull Power < 0 in trending down (ADX rising)
# - In ranging market (ADX<20): fade extremes - long when Bull Power < -0.5*ATR, short when Bear Power < -0.5*ATR
# - Volume confirmation: current volume > 1.2x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX for regime filter (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (14-period)
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h indicators
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema_13
    bear_power = ema_13 - close
    
    # ATR for volatility (14-period)
    tr_6h = np.maximum(np.abs(high[1:] - low[1:]), 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_14 = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        # Regime filters
        trending_market = adx_aligned[i] > 25
        ranging_market = adx_aligned[i] < 20
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Elder Ray turns bearish or trend changes
            if bull_power[i] <= 0 and bear_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Elder Ray turns bullish or trend changes
            if bear_power[i] <= 0 and bull_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            if volume_confirmed:
                if trending_market:
                    # Trending: follow Elder Ray direction
                    if bull_power[i] > 0 and bear_power[i] < 0:
                        position = 1
                        signals[i] = position_size
                    elif bear_power[i] > 0 and bull_power[i] < 0:
                        position = -1
                        signals[i] = -position_size
                elif ranging_market:
                    # Ranging: mean reversion at extremes
                    if bull_power[i] < -0.5 * atr_14[i]:
                        position = 1
                        signals[i] = position_size
                    elif bear_power[i] < -0.5 * atr_14[i]:
                        position = -1
                        signals[i] = -position_size
    
    return signals