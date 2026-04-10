#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d Williams %R extreme reading and volume confirmation
# - Long when price breaks above Donchian high AND 1d Williams %R < -80 (oversold) AND 1d volume > 1.3x 20-period volume SMA
# - Short when price breaks below Donchian low AND 1d Williams %R > -20 (overbought) AND 1d volume > 1.3x 20-period volume SMA
# - Exit: ATR(14) trailing stop (2.5x ATR from extreme) or Donchian midpoint reversion
# - Position sizing: 0.25 discrete level to control drawdown and reduce fee churn
# - Target: 30-60 trades/year on 4h timeframe to stay within fee drag limits
# - Williams %R provides mean-reversion edge in ranging markets while Donchian captures trends
# - Volume confirmation ensures breakouts have participation
# - ATR stoploss manages risk during volatile periods

name = "4h_donchian_williamsr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 20-period Donchian channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ATR(14) for stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 1d Williams %R (14-period)
    lookback = 14
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d + 1e-10)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Track extreme prices for trailing stop
    long_extreme = np.full(n, np.nan)
    short_extreme = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(atr[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.3x 20-period volume SMA
        # Get 1d volume for current 4h bar (each 1d bar = 6 4h bars)
        idx_1d = i // 6
        if idx_1d < len(volume_1d):
            vol_confirm = volume_1d[idx_1d] > 1.3 * volume_sma_20_1d_aligned[i]
        else:
            vol_confirm = False
        
        # Williams %R conditions
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # Update extremes for trailing stop
        if position == 1:  # Long position
            if np.isnan(long_extreme[i-1]):
                long_extreme[i] = close[i]
            else:
                long_extreme[i] = max(long_extreme[i-1], close[i])
        elif position == -1:  # Short position
            if np.isnan(short_extreme[i-1]):
                short_extreme[i] = close[i]
            else:
                short_extreme[i] = min(short_extreme[i-1], close[i])
        else:
            long_extreme[i] = np.nan
            short_extreme[i] = np.nan
        
        # ATR-based trailing stop conditions (2.5x ATR)
        stop_long = False
        stop_short = False
        
        if position == 1 and not np.isnan(long_extreme[i]):
            stop_long = close[i] < long_extreme[i] - 2.5 * atr[i]
        elif position == -1 and not np.isnan(short_extreme[i]):
            stop_short = close[i] > short_extreme[i] + 2.5 * atr[i]
        
        # Donchian midpoint reversion exit
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        if position == 0:  # Flat - look for entry
            if breakout_up and williams_oversold and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and williams_overbought and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if stop_long or exit_long:
                position = 0
                signals[i] = 0.0
                long_extreme[i] = np.nan  # Reset extreme
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if stop_short or exit_short:
                position = 0
                signals[i] = 0.0
                short_extreme[i] = np.nan  # Reset extreme
            else:
                signals[i] = -0.25
    
    return signals