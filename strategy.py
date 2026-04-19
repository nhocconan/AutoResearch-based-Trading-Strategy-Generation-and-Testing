#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h/1d trend filters and volume confirmation
# Uses 1h as primary timeframe with 4h trend filter (EMA34) and 1d trend filter (EMA89)
# Long when: price breaks above 1h Donchian upper band (20) AND price > 4h EMA34 AND price > 1d EMA89 AND volume > 1.5x 20-period average
# Short when: price breaks below 1h Donchian lower band (20) AND price < 4h EMA34 AND price < 1d EMA89 AND volume > 1.5x 20-period average
# Volume confirmation ensures institutional participation in breakouts
# Target: 15-37 trades/year per symbol (60-150 total over 4 years) for 1h timeframe
# Session filter: 08-20 UTC to reduce noise trades

name = "1h_Donchian_4h1dTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for EMA89 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1d EMA89 for trend filter
    ema_89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 89)  # Need Donchian, EMA34, EMA89 data
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(high_roll.iloc[i]) or np.isnan(low_roll.iloc[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_89_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = high_roll.iloc[i]
        lower_band = low_roll.iloc[i]
        ema_4h = ema_34_4h_aligned[i]
        ema_1d = ema_89_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above upper band AND above both EMAs AND volume confirmed
            if price > upper_band and price > ema_4h and price > ema_1d and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below lower band AND below both EMAs AND volume confirmed
            elif price < lower_band and price < ema_4h and price < ema_1d and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower band OR below either EMA
            if price < lower_band or price < ema_4h or price < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price breaks above upper band OR above either EMA
            if price > upper_band or price > ema_4h or price > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals