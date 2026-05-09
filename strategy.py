# #!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses Donchian channel breakouts (20-period high/low) on 12h chart for directional signals.
# Long when price breaks above 20-period high with volume and 1d EMA50 up.
# Short when price breaks below 20-period low with volume and 1d EMA50 down.
# Designed to capture trends in both bull and bear markets by following 1d EMA50.
# Donchian breakouts work well in trending markets and the 1d EMA50 filter avoids
# counter-trend trades during strong trends. Volume confirmation reduces false breakouts.
name = "12h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channel: 20-period high/low
    # Use pandas rolling for efficiency with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + 1d EMA up
            if (price > donchian_high[i] and 
                vol_confirm[i] and price > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + 1d EMA down
            elif (price < donchian_low[i] and 
                  vol_confirm[i] and price < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or price below 1d EMA
            if price < donchian_low[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or price above 1d EMA
            if price > donchian_high[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals