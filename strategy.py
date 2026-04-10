#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation
# - Primary: 6h timeframe for lower frequency (12-37 trades/year target)
# - HTF: 1w for major trend direction (avoid counter-trend trades in bear markets)
# - Long: Price breaks above 6h Donchian H20 + 1w close > 1w EMA200 + volume > 1.5x 20-period 6h MA
# - Short: Price breaks below 6h Donchian L20 + 1w close < 1w EMA200 + volume > 1.5x 20-period 6h MA
# - Exit: Price reverts to 6h Donchian midpoint (mean reversion) or breaks opposite H/L25
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: 1w EMA200 filter ensures we only trade with major trend, reducing whipsaws in ranging/bear markets

name = "6h_1w_donchian_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_max_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Calculate 1w EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 6h volume moving average (20-period) for volume confirmation
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup period (need 200 for 1w EMA200)
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_200_1w[i // (7*4)]) or  # 7 days * 4 six-hour bars per day = 28 bars per week
            np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get 1w trend (using last completed 1w bar)
        # 1w index corresponding to current 6h bar (approximate)
        idx_1w = i // (7 * 4)  # 28 six-hour bars in a week
        if idx_1w >= len(close_1w):
            signals[i] = 0.0
            continue
            
        # 1w trend conditions
        uptrend_1w = close_1w[idx_1w] > ema_200_1w[idx_1w]
        downtrend_1w = close_1w[idx_1w] < ema_200_1w[idx_1w]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period MA
        volume_spike = volume_6h[i] > 1.5 * volume_ma_20_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian H20 + 1w uptrend + volume spike
            if (close_6h[i] > high_max_20[i] and uptrend_1w and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian L20 + 1w downtrend + volume spike
            elif (close_6h[i] < low_min_20[i] and downtrend_1w and volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Calculate Donchian channels with 25-period for wider exit bands
            high_max_25 = pd.Series(high_6h).rolling(window=25, min_periods=25).max().values[i]
            low_min_25 = pd.Series(low_6h).rolling(window=25, min_periods=25).min().values[i]
            
            if position == 1:  # Long position
                exit_condition = (
                    close_6h[i] < donchian_mid[i] or  # Reverted to midpoint
                    close_6h[i] > high_max_25         # Break above H25 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_6h[i] > donchian_mid[i] or  # Reverted to midpoint
                    close_6h[i] < low_min_25          # Break below L25 (take profit)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals