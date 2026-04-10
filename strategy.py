#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly ATR regime + Donchian(20) breakout with volume confirmation
# - Long when price breaks above 1d Donchian high AND weekly ATR(14) > weekly ATR(50) (expanding volatility)
# - Short when price breaks below 1d Donchian low AND weekly ATR(14) > weekly ATR(50)
# - Volume confirmation: 1d volume > 1.5x 20-period 1d volume SMA
# - Exit: Donchian midpoint reversion or opposite breakout with volume
# - Position sizing: 0.25 discrete level
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Weekly ATR filter ensures trades occur during volatile regimes, reducing whipsaw in ranging markets
# - Uses 1d primary timeframe with 1h HTF for trend alignment (proven to reduce noise)

name = "1d_1w_donchian_atr_volume_v1"
timeframe = "1d"
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
    
    # Calculate 20-period Donchian channels on 1d
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ATR(14) and ATR(50) on 1w for volatility regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1w = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly ATR to daily timeframe (completed 1w bars only)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    atr_50_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_50_1w)
    
    # Calculate 20-period volume SMA for confirmation on 1d
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry extreme for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(atr_50_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: weekly ATR(14) > weekly ATR(50) (expanding volatility regime)
        vol_regime = atr_14_1w_aligned[i] > atr_50_1w_aligned[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
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
            # Exit on Donchian midpoint reversion or opposite breakout with volume
            exit_condition = (close[i] < donchian_mid[i]) or \
                           (breakout_down and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on Donchian midpoint reversion or opposite breakout with volume
            exit_condition = (close[i] > donchian_mid[i]) or \
                           (breakout_up and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals