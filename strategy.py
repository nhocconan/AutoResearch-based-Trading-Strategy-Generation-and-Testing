#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price action with weekly trend filter and volume confirmation.
# Long when: Price > weekly EMA200, closes above daily Donchian upper (20), and volume > 1.5x 20-day average
# Short when: Price < weekly EMA200, closes below daily Donchian lower (20), and volume > 1.5x 20-day average
# Exit when price crosses back to weekly EMA200 or volume drops below average
# Designed for low-frequency, high-conviction trades (~10-20 per year) to minimize fee drag.
# Works in bull (trend following) and bear (counter-trend reversals at extremes).
name = "1d_Donchian20_WeeklyEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for weekly EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1w_200_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_ema = ema_1w_200_aligned[i]
        donchian_upper = highest_high[i]
        donchian_lower = lowest_low[i]
        vol_ma = vol_ma_20[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price above weekly EMA200, breaks above Donchian upper, volume spike
            if price > weekly_ema and price > donchian_upper and vol_current > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA200, breaks below Donchian lower, volume spike
            elif price < weekly_ema and price < donchian_lower and vol_current > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below weekly EMA200 or volume drops below average
            if price < weekly_ema or vol_current < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above weekly EMA200 or volume drops below average
            if price > weekly_ema or vol_current < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals