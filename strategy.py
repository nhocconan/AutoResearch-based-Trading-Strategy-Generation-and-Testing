#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# In bull regime (price > 1w EMA50), go long on breakout above upper Donchian with volume spike.
# In bear regime (price < 1w EMA50), go short on breakdown below lower Donchian with volume spike.
# Uses 1w EMA50 for regime filter, daily Donchian channels for structure, and daily volume spike for confirmation.
# Designed for 30-100 total trades over 4 years (7-25/year) with focus on BTC/ETH as primary symbols.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (prior completed 1w bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data for Donchian channel calculation (prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate prior completed 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel = max(high, 20)
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel = min(low, 20)
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d (wait for 1d bar to complete)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Calculate volume regime: current 1d volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1w EMA50, bear if close < 1w EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: breakout above upper Donchian with volume spike
            long_entry = (close_val > upper_val) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: breakdown below lower Donchian with volume spike
            short_entry = (close_val < lower_val) and vol_spike
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
            # Exit on breakdown below lower Donchian (failure of bullish breakout) or regime change to bear
            if close_val < lower_val or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on breakout above upper Donchian (failure of bearish breakdown) or regime change to bull
            if close_val > upper_val or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals