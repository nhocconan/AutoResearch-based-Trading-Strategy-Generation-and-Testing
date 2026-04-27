#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with 1w trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes strong moves. Breakout from squeeze
# with volume and higher timeframe trend captures momentum. Works in bull/bear by
# filtering breakout direction with 1w EMA trend. Target: 30-100 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Bollinger Bands (20, 2) on 1d
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean()
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = (sma_20 + bb_std * std_20).values
    bb_lower = (sma_20 - bb_std * std_20).values
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width below 20-period average of width
    bb_width_ma = pd.Series(bb_width).rolling(window=bb_period, min_periods=bb_period).mean()
    squeeze = bb_width < bb_width_ma.values
    
    # 1w EMA trend filter (34-period)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 1.5 x 20-period average (20 days)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (20 for BB, 20 for BB width MA, 20 for vol MA)
    start_idx = max(20, 20, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(squeeze[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1w EMA
        bullish_trend = price > ema_34_1w_aligned[i]
        bearish_trend = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: break above upper BB with volume, bullish trend, and squeeze
            if price > bb_upper[i] and vol_filter and bullish_trend and squeeze[i]:
                signals[i] = size
                position = 1
            # Short: break below lower BB with volume, bearish trend, and squeeze
            elif price < bb_lower[i] and vol_filter and bearish_trend and squeeze[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or trend turns bearish
            bb_middle = (bb_upper[i] + bb_lower[i]) / 2
            if price <= bb_middle or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle BB or trend turns bullish
            bb_middle = (bb_upper[i] + bb_lower[i]) / 2
            if price >= bb_middle or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Bollinger_Squeeze_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0