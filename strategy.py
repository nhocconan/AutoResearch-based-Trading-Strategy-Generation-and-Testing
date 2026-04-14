# 1. Hypothesis:
# This strategy implements a 4-hour Bollinger Band squeeze breakout with 1-day RSI trend filter and volume confirmation.
# It aims to capture explosive moves after periods of low volatility, using Bollinger Band width to detect squeeze conditions.
# The 1-day RSI filter ensures trades align with the higher timeframe trend, reducing whipsaws in choppy markets.
# Volume confirmation ensures breakouts are supported by institutional participation.
# Exit occurs when price reverts to the Bollinger Band middle (20-period SMA).
# Designed to work in both bull and bear markets by focusing on volatility breakouts rather than directional bias.
# Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.

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
    
    # Load 1d data ONCE before loop for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands (20-period, 2 standard deviations)
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band width percentile (50-period) to detect squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    
    # Calculate RSI on 1d (14-period) for trend filter
    rsi_period = 14
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get RSI values aligned to 4h timeframe
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
        rsi_val = rsi_1d_aligned[i]
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Bollinger squeeze condition: BB width below 20th percentile (low volatility)
        squeeze_condition = bb_width_percentile[i] < 20
        
        if position == 0:
            # Long setup: Bollinger breakout above upper band AND RSI > 50 (bullish bias) AND volume confirmation
            if (price > bb_upper[i] and rsi_val > 50 and vol > vol_threshold and squeeze_condition):
                position = 1
                signals[i] = position_size
            # Short setup: Bollinger breakout below lower band AND RSI < 50 (bearish bias) AND volume confirmation
            elif (price < bb_lower[i] and rsi_val < 50 and vol > vol_threshold and squeeze_condition):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price reverts to Bollinger Band middle (20-period SMA)
            if price < sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price reverts to Bollinger Band middle (20-period SMA)
            if price > sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_Squeeze_RSI_Volume"
timeframe = "4h"
leverage = 1.0