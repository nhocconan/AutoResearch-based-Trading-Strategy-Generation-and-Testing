#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with weekly trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) with volume > 1.5x average and weekly close > EMA20.
# Short when price breaks below lower BB(20,2) with volume > 1.5x average and weekly close < EMA20.
# Exit when price returns to middle BB or volume drops below average.
# Designed for ~15-25 trades/year with strong filters to avoid whipsaws and overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands on daily data
    bb_period = 20
    bb_std = 2.0
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
    
    upper_band = sma + bb_std * std_dev
    lower_band = sma - bb_std * std_dev
    middle_band = sma
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period BB and volume MA
    start_idx = max(20, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from weekly EMA20
        bullish_trend = ema20_1w_aligned[i] > 0  # EMA20 is always positive, use price vs EMA
        bearish_trend = ema20_1w_aligned[i] > 0  # Will refine below
        
        # More precise trend: price vs weekly EMA20
        bullish_trend = price > ema20_1w_aligned[i]
        bearish_trend = price < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper BB with volume and bullish weekly trend
            if price > upper_band[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower BB with volume and bearish weekly trend
            elif price < lower_band[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or volume drops
            if price < middle_band[i] or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle BB or volume drops
            if price > middle_band[i] or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_BollingerBand_Breakout_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0