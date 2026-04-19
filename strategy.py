#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h directional filter + volume confirmation
# Uses 6h as primary timeframe with 12h trend filter (EMA34) and volume spike (>1.8x average)
# Long when: price > Donchian upper band AND price > 12h EMA34 AND volume confirmed
# Short when: price < Donchian lower band AND price < 12h EMA34 AND volume confirmed
# Donchian channels provide clear breakout levels, EMA34 filters for trend alignment
# Volume confirmation ensures institutional participation in breakouts
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years)

name = "6h_Donchian20_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        ema_trend = ema_34_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Enter long: price > upper band AND above 12h EMA34 AND volume confirmed
            if price > upper_band and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price < lower band AND below 12h EMA34 AND volume confirmed
            elif price < lower_band and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price < lower band OR below 12h EMA34
            if price < lower_band or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price > upper band OR above 12h EMA34
            if price > upper_band or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals