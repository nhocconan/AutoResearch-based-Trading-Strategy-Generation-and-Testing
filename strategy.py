#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ATR filter and volume confirmation
# - Long when price breaks above 20-period daily Donchian high AND weekly ATR(14) > weekly ATR(50) (expanding volatility)
# - Short when price breaks below 20-period daily Donchian low AND weekly ATR(14) > weekly ATR(50)
# - Volume confirmation: daily volume > 1.5x 20-period daily volume SMA
# - Exit: Donchian midpoint reversion or opposite breakout with volume
# - Position sizing: 0.25 discrete level
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - ATR filter ensures trades occur during volatile regimes, reducing whipsaw in ranging markets
# - Weekly timeframe provides higher conviction trend filter for daily breakouts

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
    
    # Calculate 20-period Donchian channels (daily)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) and ATR(50) for volatility regime filter
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    w_tr1 = pd.Series(wh - wl)
    w_tr2 = pd.Series(np.abs(wh - np.roll(wc, 1)))
    w_tr3 = pd.Series(np.abs(wl - np.roll(wc, 1)))
    w_tr = pd.concat([w_tr1, w_tr2, w_tr3], axis=1).max(axis=1)
    w_atr_14 = pd.Series(w_tr).rolling(window=14, min_periods=14).mean().values
    w_atr_50 = pd.Series(w_tr).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly ATR to daily timeframe
    w_atr_14_aligned = align_htf_to_ltf(prices, df_1w, w_atr_14)
    w_atr_50_aligned = align_htf_to_ltf(prices, df_1w, w_atr_50)
    
    # Calculate 20-period volume SMA for confirmation (daily)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry extreme for exit logic
    entry_price = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i-1]) or np.isnan(donchian_low[i-1]) or
            np.isnan(w_atr_14_aligned[i]) or np.isnan(w_atr_50_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: weekly ATR(14) > weekly ATR(50) (expanding volatility regime)
        vol_regime = w_atr_14_aligned[i] > w_atr_50_aligned[i]
        
        # Volume confirmation: daily volume > 1.5x 20-period daily volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals (daily)
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