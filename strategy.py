#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band squeeze breakout on 1d timeframe with volume confirmation and 1w trend filter.
# Long when price breaks above upper BB after low volatility squeeze (BB width < 20th percentile) and price > weekly EMA50.
# Short when price breaks below lower BB after squeeze and price < weekly EMA50.
# Exit when price crosses the 20-day SMA or when BB width expands above 80th percentile (volatility expansion).
# Uses volatility contraction/expansion for explosive moves, weekly trend for direction, volume for confirmation.
# Target: 15-25 trades/year per symbol to stay within frequency limits.
name = "1d_Weekly_BB_Squeeze_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and BB squeeze calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    sma_20 = close_s.rolling(window=20, min_periods=20).mean().values
    std_20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # BB width percentile for squeeze detection (20th percentile = squeeze)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct_20 = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    bb_width_pct_80 = bb_width_series.rolling(window=50, min_periods=50).quantile(0.80).values
    
    # Volume confirmation: 1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(bb_width_pct_20[i]) or 
            np.isnan(bb_width_pct_80[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sma20 = sma_20[i]
        std20 = std_20[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        bbwidth = bb_width[i]
        bbwidth_pct20 = bb_width_pct_20[i]
        bbwidth_pct80 = bb_width_pct_80[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        ema50w = ema_50_aligned[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Squeeze condition: low volatility
        squeeze = bbwidth < bbwidth_pct20
        
        if position == 0:
            # Long entry: BB squeeze breakout above upper band with volume and weekly uptrend
            if squeeze and i > 0 and close[i-1] <= upper_bb[i-1] and price > upper and \
               volume_confirmed and price > ema50w:
                signals[i] = 0.25
                position = 1
            # Short entry: BB squeeze breakout below lower band with volume and weekly downtrend
            elif squeeze and i > 0 and close[i-1] >= lower_bb[i-1] and price < lower and \
                 volume_confirmed and price < ema50w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-day SMA or volatility expansion
            if price < sma20 or bbwidth > bbwidth_pct80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day SMA or volatility expansion
            if price > sma20 or bbwidth > bbwidth_pct80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals