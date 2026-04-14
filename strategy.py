#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily VWAP bands with volume confirmation
# - Uses daily VWAP and standard deviation to create dynamic support/resistance bands
# - Requires volume > 1.3x 24-period average for institutional confirmation
# - Trades breakouts above upper band (long) or below lower band (short)
# - Designed to capture volatility expansion with controlled frequency
# - Target: 75-200 trades over 4 years to minimize fee drag while capturing significant moves
# - Discrete position sizing (0.25) to reduce churn and manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily VWAP and standard deviation bands
    typical_price = (high_1d + low_1d + close_1d) / 3
    tp_volume = typical_price * df_1d['volume'].values
    
    # Cumulative sums for VWAP calculation
    cum_tpv = np.cumsum(tp_volume)
    cum_volume = np.cumsum(df_1d['volume'].values)
    vwap = np.divide(cum_tpv, cum_volume, out=np.zeros_like(cum_tpv), where=cum_volume!=0)
    
    # Calculate standard deviation of typical price from VWAP
    squared_diff = (typical_price - vwap) ** 2
    cum_squared_diff = np.cumsum(squared_diff * df_1d['volume'].values)
    variance = np.divide(cum_squared_diff, cum_volume, out=np.zeros_like(cum_squared_diff), where=cum_volume!=0)
    std_dev = np.sqrt(variance)
    
    # Upper and lower bands (1 standard deviation)
    upper_band = vwap + std_dev
    lower_band = vwap - std_dev
    
    # 4h volume filter: current volume > 1.3x 24-period average (1 day)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    # 4h Donchian channels (20-period) for exit signals
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Create arrays for VWAP bands alignment
    upper_band_array = upper_band
    lower_band_array = lower_band
    
    upper_band_4h = align_htf_to_ltf(prices, df_1d, upper_band_array)
    lower_band_4h = align_htf_to_ltf(prices, df_1d, lower_band_array)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(vwap[i-1]) or np.isnan(std_dev[i-1]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(upper_band_4h[i]) or np.isnan(lower_band_4h[i]):
            continue
        
        if position == 0:
            # Long: Price breaks above upper band with volume confirmation
            if (close[i] > upper_band_4h[i] and close[i-1] <= upper_band_4h[i] and 
                volume[i] > vol_ma[i] * 1.3):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower band with volume confirmation
            elif (close[i] < lower_band_4h[i] and close[i-1] >= lower_band_4h[i] and 
                  volume[i] > vol_ma[i] * 1.3):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below lower band or drops below Donchian low
            if close[i] < lower_band_4h[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above upper band or rises above Donchian high
            if close[i] > upper_band_4h[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_VWAP_Bands_Volume_Breakout"
timeframe = "4h"
leverage = 1.0