#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Bollinger Bands width regime filter with 1d Donchian breakout
# - Uses 1w Bollinger Bands width percentile to detect low volatility (squeeze) conditions
# - Enters on 1d Donchian(20) breakout in direction of 1w trend when volatility is low
# - Bollinger Bands width < 20th percentile indicates compression (pre-breakout setup)
# - 1w trend: price above/below 50-period SMA on weekly chart
# - Volume confirmation: current volume > 1.5x 20-period average
# - Designed to capture explosive moves after consolidation in both bull and bear markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_BBW_Squeeze_DonchianBreakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Bollinger Bands width and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Bollinger Bands width (20, 2)
    bb_period = 20
    bb_std = 2
    ma_20 = pd.Series(df_1w['close']).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(df_1w['close']).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = ma_20 + bb_std * std_20
    lower_bb = ma_20 - bb_std * std_20
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Bands width percentile (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # 1w trend: 50-period SMA
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    
    # Align 1w indicators to 12h timeframe
    bb_width_percentile_12h = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    sma_50_1w_12h = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20 period)
    donchian_period = 20
    upper_dc = pd.Series(df_1d['high']).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_dc = pd.Series(df_1d['low']).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align 1d Donchian channels to 12h timeframe
    upper_dc_12h = align_htf_to_ltf(prices, df_1d, upper_dc)
    lower_dc_12h = align_htf_to_ltf(prices, df_1d, lower_dc)
    
    # Volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(bb_width_percentile_12h[i]) or np.isnan(sma_50_1w_12h[i]) or
            np.isnan(upper_dc_12h[i]) or np.isnan(lower_dc_12h[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Low volatility squeeze condition: BB width < 20th percentile
            squeeze_condition = bb_width_percentile_12h[i] < 20
            
            if squeeze_condition:
                # Determine 1w trend direction
                uptrend = close[i] > sma_50_1w_12h[i]
                downtrend = close[i] < sma_50_1w_12h[i]
                
                # Long entry: Donchian breakout above upper band in uptrend
                if uptrend and close[i] > upper_dc_12h[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short entry: Donchian breakout below lower band in downtrend
                elif downtrend and close[i] < lower_dc_12h[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches lower Donchian band or volatility expands
            if close[i] < lower_dc_12h[i]:
                signals[i] = 0.0
                position = 0
            elif bb_width_percentile_12h[i] > 80:  # Volatility expansion exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches upper Donchian band or volatility expands
            if close[i] > upper_dc_12h[i]:
                signals[i] = 0.0
                position = 0
            elif bb_width_percentile_12h[i] > 80:  # Volatility expansion exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals