#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 6h
# - Regime: 1d ADX < 20 (range) for mean reversion, ADX > 25 (trend) for momentum
# - In range (ADX<20): fade extreme Elder Ray (long when Bear Power < -0.5*ATR, short when Bull Power > 0.5*ATR)
# - In trend (ADX>25): follow Elder Ray (long when Bull Power > 0.5*ATR, short when Bear Power < -0.5*ATR)
# - Volume confirmation: 6h volume > 1.5x 20-period volume SMA
# - Position sizing: 0.25 discrete
# - Works in bull/bear: adapts to regime, avoids whipsaw in strong trends, captures reversals in ranges

name = "6h_1d_elderray_regime_volume_v1"
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
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h ATR(14) for power thresholds
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    atr = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume SMA(20) for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1_1d = np.maximum(df_1d_high - df_1d_low, 
                        np.maximum(np.abs(df_1d_high - np.roll(df_1d_close, 1)), 
                                   np.abs(df_1d_low - np.roll(df_1d_close, 1))))
    tr1_1d[0] = df_1d_high[0] - df_1d_low[0]
    # Plus Directional Movement
    plus_dm_1d = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low), 
                          np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    plus_dm_1d[0] = 0
    # Minus Directional Movement
    minus_dm_1d = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)), 
                           np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    minus_dm_1d[0] = 0
    # Smoothed values
    atr_1d = pd.Series(tr1_1d).rolling(window=14, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    plus_di_1d = np.where(atr_1d == 0, 0, plus_di_1d)
    minus_di_1d = np.where(atr_1d == 0, 0, minus_di_1d)
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = np.where(np.isnan(adx_1d) | np.isinf(adx_1d), 0, adx_1d)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Regime filters
        ranging_market = adx_1d_aligned[i] < 20   # ADX < 20 = range/low trend
        trending_market = adx_1d_aligned[i] > 25  # ADX > 25 = strong trend
        
        # Elder Ray power thresholds
        bull_threshold = 0.5 * atr[i]
        bear_threshold = -0.5 * atr[i]
        
        # Signals based on regime
        if ranging_market:
            # In range: fade extreme Elder Ray (mean reversion)
            long_signal = bear_power[i] < bear_threshold and vol_confirm
            short_signal = bull_power[i] > bull_threshold and vol_confirm
        elif trending_market:
            # In trend: follow Elder Ray (momentum)
            long_signal = bull_power[i] > bull_threshold and vol_confirm
            short_signal = bear_power[i] < bear_threshold and vol_confirm
        else:
            # Transition regime (ADX 20-25): no trade
            long_signal = False
            short_signal = False
        
        # Exit conditions: power returns to neutral zone
        long_exit = bull_power[i] > -0.1 * atr[i]  # Exit long when bull power improves
        short_exit = bear_power[i] < 0.1 * atr[i]   # Exit short when bear power improves
        
        if position == 0:  # Flat - look for entry
            if long_signal:
                position = 1
                signals[i] = 0.25
            elif short_signal:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals