#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period high AND close > EMA50(1d) AND volume > 1.5x 20-bar avg
# Short when price breaks below 20-period low AND close < EMA50(1d) AND volume > 1.5x 20-bar avg
# Exit when price returns to 10-period midpoint OR volume drops below average
# Target: 12-37 trades/year via tight breakout conditions + regime filter
# Works in bull markets via long breakouts, bear markets via short breakouts
# Uses 1d EMA50 for higher-timeframe trend alignment to reduce whipsaw

name = "12h_Donchian20_1dEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need sufficient history for Donchian(20) and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_50 = ema_50_1d_aligned[i]
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        midpoint = donchian_mid[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > upper band AND above 1d EMA50 AND volume confirmation
            if price > upper and price > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short breakout: price < lower band AND below 1d EMA50 AND volume confirmation
            elif price < lower and price < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to midpoint OR volume drops
            if price <= midpoint or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to midpoint OR volume drops
            if price >= midpoint or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals