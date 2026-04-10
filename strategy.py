#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1d ATR(14) > 1d ATR(50) (expanding daily volatility)
# - Short when price breaks below 20-period Donchian low AND 1d ATR(14) > 1d ATR(50)
# - Volume confirmation: 4h volume > 1.8x 20-period 4h volume SMA (stricter to reduce trades)
# - Exit: Donchian midpoint reversion
# - Position sizing: 0.25 discrete level
# - Uses 1d ATR for regime filter to avoid intrabar noise and improve BTC/ETH performance
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_donchian_1datr_volume_v1"
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
    
    # Calculate 20-period Donchian channels (4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Multi-timeframe: get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    tr1_1d = df_1d['high'] - df_1d['low']
    tr2_1d = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3_1d = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR to 4h timeframe (waits for completed 1d bar)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate 20-period volume SMA for confirmation (4h)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_50_1d_aligned[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR(14) > 1d ATR(50) (expanding daily volatility regime)
        vol_regime = atr_14_1d_aligned[i] > atr_50_1d_aligned[i]
        
        # Volume confirmation: 4h volume > 1.8x 20-period volume SMA (stricter threshold)
        vol_confirm = volume[i] > 1.8 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_regime and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_regime and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on Donchian midpoint reversion
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals