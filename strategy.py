#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with weekly trend filter and volume confirmation.
# Uses Ichimoku from daily timeframe (Tenkan/Kijun/Senkou) for trend direction and cloud filter.
# Weekly trend filter (price vs weekly Kumo) avoids counter-trend trades in strong trends.
# Volume confirmation ensures breakouts have conviction.
# Designed for 15-35 trades/year to avoid fee drag while capturing major trends.
# Works in bull/bear by only trading in direction of weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Ichimoku components for trend filter
    period9_high_w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low_w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_w = (period9_high_w + period9_low_w) / 2
    
    period26_high_w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_w = (period26_high_w + period26_low_w) / 2
    
    senkou_a_w = ((tenkan_w + kijun_w) / 2)
    period52_high_w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_w = ((period52_high_w + period52_low_w) / 2)
    
    # Align weekly Ichimoku to 6m timeframe
    tenkan_w_6h = align_htf_to_ltf(prices, df_1w, tenkan_w)
    kijun_w_6h = align_htf_to_ltf(prices, df_1w, kijun_w)
    senkou_a_w_6h = align_htf_to_ltf(prices, df_1w, senkou_a_w)
    senkou_b_w_6h = align_htf_to_ltf(prices, df_1w, senkou_b_w)
    
    # Calculate 6h ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 20)  # need Ichimoku components, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(tenkan_w_6h[i]) or np.isnan(kijun_w_6h[i]) or
            np.isnan(senkou_a_w_6h[i]) or np.isnan(senkou_b_w_6h[i]) or
            np.isnan(atr_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou A/B)
        upper_cloud = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        lower_cloud = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Weekly cloud boundaries for trend filter
        upper_cloud_w = np.maximum(senkou_a_w_6h[i], senkou_b_w_6h[i])
        lower_cloud_w = np.minimum(senkou_a_w_6h[i], senkou_b_w_6h[i])
        
        # Ichimoku signals
        tk_cross_bull = tenkan_6h[i] > kijun_6h[i]  # Bullish TK cross
        tk_cross_bear = tenkan_6h[i] < kijun_6h[i]  # Bearish TK cross
        
        # Price above/below cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Weekly trend filter: price vs weekly cloud
        weekly_uptrend = close[i] > upper_cloud_w
        weekly_downtrend = close[i] < lower_cloud_w
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: bullish TK cross + price above cloud + weekly uptrend + volume
            if (tk_cross_bull and price_above_cloud and 
                weekly_uptrend and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish TK cross + price below cloud + weekly downtrend + volume
            elif (tk_cross_bear and price_below_cloud and 
                  weekly_downtrend and vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below cloud or TK cross turns bearish
            if close[i] < lower_cloud or tk_cross_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above cloud or TK cross turns bullish
            if close[i] > upper_cloud or tk_cross_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0