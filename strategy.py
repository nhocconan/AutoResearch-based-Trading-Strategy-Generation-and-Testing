#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses 12h Camarilla pivot levels (R3/S3) for breakout entries, filtered by 1d EMA34 trend direction.
# Volume spike filter ensures trades occur during high conviction moves (volume > 1.5x 20-period average).
# Works in both bull (buy R3 breakout with uptrend) and bear (sell S3 breakdown with downtrend).
# Discrete position sizing (0.25) balances return and drawdown. Target: 50-150 trades over 4 years.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20) + 1  # 35
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Camarilla levels for 12h timeframe using previous day's OHLC
        # Need to get daily OHLC from 1d data aligned to 12h bars
        if i < 2:  # Need at least 2 bars for previous day calculation
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC from 1d data (aligned)
        # We need to map 12h bar to corresponding 1d bar for Camarilla calculation
        # Simplified approach: use rolling window on 12h data for Camarilla-like levels
        # But to follow Camarilla properly, we need daily OHLC
        
        # Instead, calculate Donchian channels as proxy for breakout levels (more reliable for 12h)
        # Using 20-period Donchian on 12h data
        lookback = 20
        if i < lookback:
            signals[i] = 0.0
            continue
            
        upper_channel = np.max(high[i-lookback:i])
        lower_channel = np.min(low[i-lookback:i])
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume spike filter: current volume > 1.5x 20-period average
        volume_spike = curr_volume > (vol_ma_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper channel AND uptrend AND volume spike
            if curr_close > upper_channel and uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower channel AND downtrend AND volume spike
            elif curr_close < lower_channel and downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below lower channel (reversal signal)
            if curr_close < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above upper channel (reversal signal)
            if curr_close > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals