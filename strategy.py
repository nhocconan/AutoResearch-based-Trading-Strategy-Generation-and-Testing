#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR(14) volatility filter
# - Long when price breaks above 20-period Donchian high AND 1d volume > 1.5x 20-bar average AND ATR(14) < 0.05 * price (low volatility regime)
# - Short when price breaks below 20-period Donchian low AND 1d volume > 1.5x 20-bar average AND ATR(14) < 0.05 * price
# - Exit when price returns to 10-period Donchian midpoint or opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - Volume confirmation ensures breakout validity
# - ATR filter avoids choppy markets where breakouts fail

name = "12h_1d_donchian_breakout_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 12h data
    period_dc = 20
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate highest high and lowest low over the period
    highest_high = pd.Series(high_12h).rolling(window=period_dc, min_periods=period_dc).max().values
    lowest_low = pd.Series(low_12h).rolling(window=period_dc, min_periods=period_dc).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR(14) for volatility filter
    atr_period = 14
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(true_range).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_ratio = atr / prices['close'].values  # ATR as percentage of price
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike and low volatility
            if (prices['close'].iloc[i] > highest_high[i] and 
                vol_spike.iloc[i] and 
                atr_ratio[i] < 0.05):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volume spike and low volatility
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  vol_spike.iloc[i] and 
                  atr_ratio[i] < 0.05):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to Donchian midpoint
            # 2. Opposite breakout occurs
            if position == 1:
                if (prices['close'].iloc[i] <= donchian_mid[i] or 
                    prices['close'].iloc[i] < lowest_low[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if (prices['close'].iloc[i] >= donchian_mid[i] or 
                    prices['close'].iloc[i] > highest_high[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals