#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses weekly EMA50 for trend direction, daily Donchian channels for breakout signals
# Volume confirmation ensures institutional participation and reduces false breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking breakouts aligned with weekly trend
# ATR-based stoploss implemented via signal=0 when price crosses opposite Donchian level

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Load daily data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50  # max(50 for weekly EMA, 20 for Donchian) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of weekly EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volatility filter: avoid extremely low volatility chop
        atr_ma = pd.Series(atr_14_aligned).rolling(window=10, min_periods=10).mean().shift(1).values
        if np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            volatility_filter = True
        else:
            volatility_filter = atr_14_aligned[i] > (atr_ma[i] * 0.7)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND uptrend AND volume confirm AND volatility filter
            if (close[i] > donchian_high_aligned[i] and 
                uptrend and 
                volume_confirm[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND downtrend AND volume confirm AND volatility filter
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend and 
                  volume_confirm[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend reverses
            if (close[i] < donchian_low_aligned[i] or 
                not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend reverses
            if (close[i] > donchian_high_aligned[i] or 
                not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals