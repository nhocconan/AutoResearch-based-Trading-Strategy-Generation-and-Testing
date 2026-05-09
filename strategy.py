# %%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d VWAP trend filter and volume confirmation
# Donchian breakouts capture momentum, VWAP on 1d filters trend direction (price > VWAP = bullish),
# and volume > 2x 20-period average confirms institutional participation.
# Works in bull/bear markets by requiring trend alignment. Target: 20-50 trades over 4 years.
name = "4h_Donchian20_1dVWAP_Trend_Volume"
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d VWAP (Volume Weighted Average Price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_num = (typical_price * df_1d['volume']).cumsum()
    vwap_den = df_1d['volume'].cumsum()
    vwap_1d = (vwap_num / vwap_den).values
    
    # Align VWAP to 4h
    vwap_4h = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian and VWAP calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_4h[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above upper band
        short_breakout = close[i] < donchian_low[i-1]  # Break below lower band
        
        trend_up = close[i] > vwap_4h[i]
        trend_down = close[i] < vwap_4h[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if long_breakout and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif short_breakout and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout below lower band or trend reversal
            if close[i] < donchian_low[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above upper band or trend reversal
            if close[i] > donchian_high[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
# %%