#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA34) and volume confirmation.
# Long when: price breaks above Donchian high(20), price > EMA34(12h), volume > 1.5x 20-period average
# Short when: price breaks below Donchian low(20), price < EMA34(12h), volume > 1.5x 20-period average
# Exit when price returns to Donchian midpoint (mean reversion within the channel)
# Donchian captures breakouts, EMA34 filters trend direction, volume confirms breakout strength.
# Target: 20-40 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).

name = "4h_Donchian20_EMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h data
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation (longest indicator)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        ema34 = ema34_12h_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        mid_channel = donchian_mid[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian band, above EMA34, volume spike
            if (high_price > upper_channel and price > ema34 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian band, below EMA34, volume spike
            elif (low_price < lower_channel and price < ema34 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian midpoint (mean reversion)
            if price <= mid_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian midpoint (mean reversion)
            if price >= mid_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals