#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume Spike + 1w/1d Regime Filter
# Uses 1w EMA50 for long-term trend and 1d ADX for short-term regime (trending/ranging)
# In trending markets (1w EMA50 up + 1d ADX>25): enter on 6h volume spikes (2x 20-period avg) in trend direction
# In ranging markets (1w EMA50 flat + 1d ADX<20): fade extreme price moves using 6h RSI(14) <30/>70
# Volume confirmation ensures participation, regime filter avoids wrong-side trades
# Discrete position sizing 0.25 minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime via multi-timeframe analysis

name = "6h_VolumeSpike_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for long-term trend
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d ADX for short-term regime
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 6h RSI for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60  # max(20 for volume, 50 for 1w EMA, 34 for ADX) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Determine long-term trend from 1w EMA50
        if i > start_idx:
            ema50_prev = ema50_1w_aligned[i-1]
            ema50_curr = ema50_1w_aligned[i]
            trend_up = ema50_curr > ema50_prev
            trend_down = ema50_curr < ema50_prev
        else:
            trend_up = trend_down = False
        
        # Determine short-term regime from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if trending and trend_up:
                # In uptrend: long on volume spike
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            elif trending and trend_down:
                # In downtrend: short on volume spike
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif ranging:
                # In ranging market: mean reversion using RSI extremes
                if rsi[i] < 30 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending and trend_up:
                # Exit long when trend weakens or volume dries up
                if not trend_up or not volume_spike[i]:
                    exit_signal = True
            elif ranging:
                # Exit long when RSI reaches overbought or mean reversion complete
                if rsi[i] > 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending and trend_down:
                # Exit short when trend weakens or volume dries up
                if not trend_down or not volume_spike[i]:
                    exit_signal = True
            elif ranging:
                # Exit short when RSI reaches oversold or mean reversion complete
                if rsi[i] < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals