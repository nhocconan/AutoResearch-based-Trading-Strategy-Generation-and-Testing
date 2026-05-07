# 4h_Bollinger_Band_Momentum_1dTrend_Confirmation_v1
# This strategy identifies high-probability momentum entries by combining Bollinger Band breakouts
# with 1-day trend confirmation and volume surge filters. It focuses on capturing strong
# directional moves during trending markets while filtering out choppy conditions. Designed
# for the 4-hour timeframe with 1-day higher timeframe trend filter to ensure alignment with
# institutional momentum. The strategy uses discrete position sizing to minimize transaction
# costs and is structured to perform in both bull and bear markets by only taking trades
# in the direction of the higher timeframe trend.

name = "4h_Bollinger_Band_Momentum_1dTrend_Confirmation_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1-day data for trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Bollinger Bands (20, 2) on 4h timeframe
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_band = (sma_20 + 2 * std_20).values
    lower_band = (sma_20 - 2 * std_20).values
    
    # Volume surge: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~0.5 days (3*4h) to prevent overtrading
    
    start_idx = 20  # Ensure enough data for Bollinger Bands and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above upper Bollinger Band with volume surge in 1d uptrend
            if (close[i] > upper_band[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below lower Bollinger Band with volume surge in 1d downtrend
            elif (close[i] < lower_band[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below middle Bollinger Band (SMA20) or 1d trend changes to down
            if close[i] < sma_20.iloc[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above middle Bollinger Band (SMA20) or 1d trend changes to up
            if close[i] > sma_20.iloc[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Bollinger Band breakouts combined with 1-day trend confirmation and volume surge filters
# capture institutional momentum moves while avoiding false signals in choppy markets. The strategy
# only takes long positions in 1-day uptrends and short positions in 1-day downtrends, ensuring
# alignment with higher timeframe momentum. Bollinger Bands (20,2) provide dynamic support/resistance
# levels that adapt to volatility, while the volume surge filter (2.0x 20-period average) confirms
# institutional participation. The middle Bollinger Band (SMA20) serves as an objective exit point.
# Discrete position sizing (0.25) and cooldown periods minimize transaction costs. Designed for
# the 4-hour timeframe to balance signal frequency with reliability, targeting 50-150 total trades
# over 4 years (12-37/year) to overcome fee drag. Works in both bull markets (longs in uptrends)
# and bear markets (shorts in downtrends) by following the 1-day trend filter.