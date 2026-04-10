#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily ATR filter and volume spike confirmation
# - Long when price breaks above 20-period Donchian high AND ATR(14) > 1.5x ATR(50) (expanding volatility)
# - Short when price breaks below 20-period Donchian low AND ATR(14) > 1.5x ATR(50)
# - Volume confirmation: 4h volume > 1.8x 20-period 4h volume SMA
# - Exit: Price crosses Donchian midpoint (mean reversion within the channel)
# - Position sizing: 0.25 discrete level
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - ATR filter ensures trades only during volatile regimes, reducing whipsaw in ranging markets
# - Volume spike confirms institutional participation in breakouts

name = "4h_donchian_atr_volume_v2"
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
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) > 1.5x ATR(50) (expanding volatility regime)
        vol_filter = atr_14[i] > 1.5 * atr_50[i]
        
        # Volume confirmation: 4h volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > 1.8 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_filter and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_filter and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when price crosses Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when price crosses Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals