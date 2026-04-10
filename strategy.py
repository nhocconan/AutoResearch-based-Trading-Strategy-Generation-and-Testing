#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation
# - Primary: 1d timeframe for lower frequency (target 30-100 trades over 4 years)
# - HTF: 1w for trend direction via HMA(21) to avoid counter-trend trades
# - Long: Price breaks above Donchian(20) high + 1w HMA(21) upward + volume > 1.5x 20-day MA
# - Short: Price breaks below Donchian(20) low + 1w HMA(21) downward + volume > 1.5x 20-day MA
# - Exit: Price reverts to Donchian(20) midpoint or opposite breakout
# - Position sizing: 0.30 (discrete level)
# - Works in bull/bear: Donchian captures breakouts in trends, HMA filter avoids false signals in ranges/chops

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLCV
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian Channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1w HMA(21) for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    close_1w_series = pd.Series(close_1w)
    wma_half = close_1w_series.rolling(window=half_length, min_periods=half_length).apply(
        lambda x: wma(x, half_length), raw=False
    ).values
    wma_full = close_1w_series.rolling(window=21, min_periods=21).apply(
        lambda x: wma(x, 21), raw=False
    ).values
    
    # Handle array lengths for WMA calculation
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).rolling(window=sqrt_length, min_periods=sqrt_length).apply(
        lambda x: wma(x, sqrt_length), raw=False
    ).values
    
    # Align 1w HMA to 1d timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: 1w HMA slope (upward/downward)
        if i >= 51:
            hma_slope = hma_21_aligned[i] - hma_21_aligned[i-1]
            hma_up = hma_slope > 0
            hma_down = hma_slope < 0
        else:
            hma_up = False
            hma_down = False
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high + HMA up + volume spike
            if (close_1d[i] > donchian_high[i] and hma_up and volume_spike):
                position = 1
                signals[i] = 0.30
            # Short entry: Price breaks below Donchian low + HMA down + volume spike
            elif (close_1d[i] < donchian_low[i] and hma_down and volume_spike):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian midpoint (mean reversion)
            # 2. Price breaks opposite Donchian level (take profit/stop)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_1d[i] < donchian_mid[i] or  # Reverted to midpoint
                    close_1d[i] > donchian_high[i]    # Break above high (trailing stop)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_1d[i] > donchian_mid[i] or  # Reverted to midpoint
                    close_1d[i] < donchian_low[i]     # Break below low (trailing stop)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals