#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ATR regime filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 12h ATR(14) > 12h ATR(50) (expanding volatility)
# - Short when price breaks below 20-period Donchian low AND 12h ATR(14) > 12h ATR(50)
# - Volume confirmation: 4h volume > 1.8x 20-period 4h volume SMA (stricter to reduce trades)
# - Exit: Donchian midpoint reversion
# - Position sizing: 0.25 discrete level
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Uses 12h ATR for regime filter to avoid look-ahead and ensure completed-bar timing
# - Higher volume threshold (1.8x) to reduce overtrading seen in recent failures

name = "4h_12h_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Load 12h HTF data ONCE before loop (MANDATORY)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h ATR(14) and ATR(50) for volatility regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_50_12h = pd.Series(tr_12h).rolling(window=50, min_periods=50).mean().values
    
    # Align 12h ATR to 4h timeframe (proper completed-bar timing)
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    atr_50_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_50_12h)
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry extreme for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(atr_14_12h_aligned[i]) or np.isnan(atr_50_12h_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 12h ATR(14) > 12h ATR(50) (expanding volatility regime)
        vol_regime = atr_14_12h_aligned[i] > atr_50_12h_aligned[i]
        
        # Volume confirmation: 4h volume > 1.8x 20-period volume SMA (stricter threshold)
        vol_confirm = volume[i] > 1.8 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_regime and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            elif breakout_down and vol_regime and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals