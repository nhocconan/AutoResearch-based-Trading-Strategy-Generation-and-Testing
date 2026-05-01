#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND 1d ATR(14) < 1.5x 50-period MA of ATR (low vol regime) AND volume > 1.5x 20-bar average.
# Short when price breaks below 20-period Donchian low AND same vol regime AND volume confirmation.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years.
# Works in bull markets (trend continuation) and bear markets (breakouts in low vol regimes often precede strong moves).

name = "4h_Donchian20_1dATR_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR and MA calculations
        return np.zeros(n)
    
    # 1d ATR(14) calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 50-period MA of 1d ATR
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR and its MA to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Low volatility regime: ATR < 1.5x its 50-period MA
    low_vol_regime = atr_aligned < (atr_ma_aligned * 1.5)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for ATR MA and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr_aligned[i]) or np.isnan(atr_ma_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(low_vol_regime[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above upper band
        breakout_down = curr_low < donchian_low[i]  # break below lower band
        
        # Regime filter: only trade in low volatility environments
        in_low_vol = low_vol_regime[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND low vol regime AND volume confirmation
            if (breakout_up and 
                in_low_vol and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND low vol regime AND volume confirmation
            elif (breakout_down and 
                  in_low_vol and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian low (stoploss) OR high vol regime (change in conditions)
            if (curr_low < donchian_low[i] or 
                not in_low_vol):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high (stoploss) OR high vol regime
            if (curr_high > donchian_high[i] or 
                not in_low_vol):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals