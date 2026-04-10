#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily ATR-based volatility filter and volume spike confirmation
# - Long when price breaks above 20-period Donchian high AND ATR(14) > 1.5x ATR(50) (expanding volatility) AND volume > 1.5x 20-period volume SMA
# - Short when price breaks below 20-period Donchian low AND ATR(14) > 1.5x ATR(50) AND volume > 1.5x 20-period volume SMA
# - Exit: Donchian midpoint reversion
# - Position sizing: 0.25 discrete level
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Volatility filter ensures breakouts occur during genuine expansion, reducing false signals in ranging markets

name = "4h_donchian_volatility_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (not used in this strategy but kept for MTF template)
    # df_1d = get_htf_data(prices, '1d')
    
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
    
    # Calculate ATR for volatility filter
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = abs(pd.Series(high).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
    tr3 = abs(pd.Series(low).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry extreme for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(max(donchian_period, 50), n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) > 1.5x ATR(50) (expanding volatility)
        vol_filter = atr_14[i] > 1.5 * atr_50[i]
        
        # Volume confirmation: volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_filter and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            elif breakout_down and vol_filter and vol_confirm:
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