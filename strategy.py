#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) (measures buying/selling pressure)
# - Long: Bull Power > 0 (buying pressure) + 1d ADX > 25 (strong trend) + 1d volume > 1.5x 20-period MA
# - Short: Bear Power < 0 (selling pressure) + 1d ADX > 25 (strong trend) + 1d volume > 1.5x 20-period MA
# - Exit: Elder Power crosses zero (Bull Power <= 0 for longs, Bear Power >= 0 for shorts)
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year)
# - Elder Ray identifies institutional buying/selling pressure
# - 1d ADX > 25 ensures we only trade in strong trending markets (avoids chop)
# - Volume confirmation ensures institutional participation

name = "6h_1d_elderray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema_13_6h  # Buying pressure
    bear_power = low_6h - ema_13_6h   # Selling pressure
    
    # Calculate 1d ADX(14) for trend strength filter
    # ADX requires +DI, -DI, and TR
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d)
    dx_1d = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Handle division by zero in ADX calculation
    adx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, adx_1d)
    
    # Align 1d indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d data for current 6h bar (completed 1d bar)
        adx_current = adx_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Regime condition: strong trend (ADX > 25)
        strong_trend = adx_current > 25
        
        # Volume spike condition: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d_current > 1.5 * volume_ma_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 (buying pressure) + strong trend + volume spike
            if (bull_power[i] > 0 and strong_trend and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 (selling pressure) + strong trend + volume spike
            elif (bear_power[i] < 0 and strong_trend and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Elder Power crosses zero
            if position == 1:
                if bull_power[i] <= 0:  # Exit long when Bull Power crosses below zero
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power[i] >= 0:  # Exit short when Bear Power crosses above zero
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals