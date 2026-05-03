#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# In bull regime (price > 12h EMA50), go long on breakout above upper band with volume spike.
# In bear regime (price < 12h EMA50), go short on breakdown below lower band with volume spike.
# Uses Donchian channels from prior completed 12h for structure, 12h EMA50 for regime filter,
# and 4h volume spike for confirmation. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels (prior completed 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate prior 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h (wait for 12h bar to complete)
    dh_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 12h data for EMA50 trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        dh = dh_aligned[i]
        dl = dl_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(dh) or np.isnan(dl) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 12h EMA50, bear if close < 12h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: breakout above upper Donchian band with volume spike
            long_entry = (close_val > dh) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: breakdown below lower Donchian band with volume spike
            short_entry = (close_val < dl) and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on breakdown below lower band (failure of bullish breakout) or regime change to bear
            if close_val < dl or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on breakout above upper band (failure of bearish breakdown) or regime change to bull
            if close_val > dh or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals