#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with daily volume confirmation and weekly trend filter.
# Long when: BB width at 30-period low (squeeze), price breaks above upper band, volume > 1.5x 20-period avg, weekly close > weekly EMA20
# Short when: BB width at 30-period low, price breaks below lower band, volume > 1.5x 20-period avg, weekly close < weekly EMA20
# Exit when price crosses back inside the Bollinger Bands.
# This strategy captures volatility breakouts after low volatility periods, works in both bull and bear markets by filtering with weekly trend.
# Target: 20-40 trades/year per symbol. Uses Bollinger Bands with standard parameters (20, 2).
name = "6h_BollingerSqueeze_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Bollinger Bands and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands on 1d data (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + bb_std * std_20
    lower_bb = sma_20 - bb_std * std_20
    bb_width = upper_bb - lower_bb
    
    # Calculate BB width percentile for squeeze detection (30-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=30, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 20-period volume average for confirmation (on 1d data)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sma_val = sma_20_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        bb_width_percentile_val = bb_width_percentile_aligned[i]
        weekly_ema_val = ema20_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Squeeze condition: BB width at or below 20th percentile (low volatility)
        is_squeeze = bb_width_percentile_val <= 0.20
        
        if position == 0:
            # Long entry: squeeze + price breaks above upper BB + volume confirmation + weekly uptrend
            if (is_squeeze and price > upper_bb_val and 
                vol > 1.5 * vol_ma and price > weekly_ema_val):
                signals[i] = 0.25
                position = 1
            # Short entry: squeeze + price breaks below lower BB + volume confirmation + weekly downtrend
            elif (is_squeeze and price < lower_bb_val and 
                  vol > 1.5 * vol_ma and price < weekly_ema_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back inside Bollinger Bands (below upper band)
            if price < upper_bb_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back inside Bollinger Bands (above lower band)
            if price > lower_bb_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals