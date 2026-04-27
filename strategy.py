#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger squeeze breakout with 1w trend filter and volume confirmation
# Bollinger squeeze (low volatility) precedes explosive moves.
# Breakout above upper band with volume and bullish 1w trend = long.
# Breakdown below lower band with volume and bearish 1w trend = short.
# Uses 1w EMA for trend filter to avoid counter-trend trades.
# Target: 7-25 trades per year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2.0)
    bb_mid_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper_1d = bb_mid_1d + 2.0 * bb_std_1d
    bb_lower_1d = bb_mid_1d - 2.0 * bb_std_1d
    
    # Bollinger Band Width (squeeze indicator)
    bb_width_1d = (bb_upper_1d - bb_lower_1d) / bb_mid_1d
    
    # Squeeze condition: BB width below 20-period average (low volatility)
    bb_width_ma_20 = pd.Series(bb_width_1d).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width_1d < bb_width_ma_20
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA trend filter (34-period)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Bollinger (20), BB width MA (20), volume MA (20)
    start_idx = max(20, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_upper_1d[i]) or np.isnan(bb_lower_1d[i]) or
            np.isnan(squeeze[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1w EMA
        bullish_trend = price > ema_34_1w_aligned[i]
        bearish_trend = price < ema_34_1w_aligned[i]
        
        bb_upper = bb_upper_1d[i]
        bb_lower = bb_lower_1d[i]
        is_squeeze = squeeze[i]
        
        if position == 0:
            # Long: Bollinger squeeze breakout above upper band + volume + bullish 1w trend
            if is_squeeze and price > bb_upper and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: Bollinger squeeze breakout below lower band + volume + bearish 1w trend
            elif is_squeeze and price < bb_lower and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below middle band or trend turns bearish
            if price < bb_mid_1d[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above middle band or trend turns bullish
            if price > bb_mid_1d[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Bollinger_Squeeze_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0