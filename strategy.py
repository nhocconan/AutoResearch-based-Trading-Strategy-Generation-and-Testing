#0.10% round-trip fees. Goal: 10-30 trades/year on 1d.
#Entry: 1d close breaks weekly Donchian(20) + weekly EMA20 trend + volume spike.
#Exit: Close re-enters Donchian channel or trend flips.
#Position: 0.25 long/short.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA20 for trend filter
    ema_20 = pd.Series(df_1w['close']).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20)
    ema_trend = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Daily volume spike filter (volume > 2.0x 20-day average)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly data (20 periods) + volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_trend[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + above weekly EMA20 + volume spike
            if price_now > donchian_high[i] and price_now > ema_trend[i] and volume_spike[i]:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian low + below weekly EMA20 + volume spike
            elif price_now < donchian_low[i] and price_now < ema_trend[i] and volume_spike[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR trend flips down
            if price_now < donchian_high[i] or price_now < ema_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR trend flips up
            if price_now > donchian_low[i] or price_now > ema_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian_EMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0