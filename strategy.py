#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme reversal with 1d ADX regime filter and volume spike confirmation.
# In bear markets (ADX > 25), extreme Williams %R (< -80 for long, > -20 for short) with volume spike
# signals mean reversion retracements. In ranging markets (ADX < 20), the same extremes trigger
# continuation breaks. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 50-150 trades over 4 years.

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R(14) - momentum oscillator
    def calculate_williams_r(high, low, close, window):
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = calculate_williams_r(high, low, close, 14)
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) - trend strength
    def calculate_adx(high, low, close, window):
        plus_dm = pd.Series(high).diff()
        minus_dm = pd.Series(low).diff().copy()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr1 = pd.Series(high).diff()
        tr2 = pd.Series(low).diff()
        tr3 = pd.Series(close).diff()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
        
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=window, min_periods=window).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=window, min_periods=window).mean().values / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=window, min_periods=window).mean().values
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # warmup for Williams %R
        # Skip if missing data
        if (np.isnan(wr[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Regime-based entry logic
            if adx_aligned[i] > 25:  # Trending market (bearish bias for BTC/ETH)
                # Mean reversion: extreme oversold/overbought with volume spike
                if (wr[i] < -80 and  # Extreme oversold
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                elif (wr[i] > -20 and  # Extreme overbought
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market (ADX < 25)
                # Continuation: extreme reading suggests momentum exhaustion before breakout
                if (wr[i] < -80 and  # Oversold - potential bullish continuation
                    volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                elif (wr[i] > -20 and  # Overbought - potential bearish continuation
                      volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: Williams %R returns to neutral territory OR extreme overbought
            if wr[i] > -50 or wr[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to neutral territory OR extreme oversold
            if wr[i] < -50 or wr[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals