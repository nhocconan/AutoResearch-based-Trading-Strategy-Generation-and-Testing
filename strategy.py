#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 12h EMA trend filter and volume confirmation
# Long when BB width < 20th percentile (squeeze) and price breaks above upper BB with volume > 1.5x 20-bar average and price > 12h EMA(50)
# Short when BB width < 20th percentile (squeeze) and price breaks below lower BB with volume > 1.5x 20-bar average and price < 12h EMA(50)
# Exit when price returns to middle BB or opposite BB break occurs
# Uses 4h timeframe targeting 19-50 trades/year (~75-200 total over 4 years) to minimize fee drag.
# Works in bull markets via upward breakouts and in bear markets via downward breakouts.

name = "4h_BollingerSqueeze_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (20-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # BB(20) and volume MA(20) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        bb_width_regime = bb_width_percentile[i] < 0.20  # Squeeze regime: BB width < 20th percentile
        bb_mid = bb_middle[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: BB squeeze + price breaks above upper BB + volume spike + price > 12h EMA50
            if bb_width_regime and price > bb_up and vol_confirm and price > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: BB squeeze + price breaks below lower BB + volume spike + price < 12h EMA50
            elif bb_width_regime and price < bb_low and vol_confirm and price < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on return to middle BB or short breakout
            # Exit on price return to middle BB or short breakout (price < lower BB) with volume confirmation
            if price < bb_mid or (price < bb_low and vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on return to middle BB or long breakout
            # Exit on price return to middle BB or long breakout (price > upper BB) with volume confirmation
            if price > bb_mid or (price > bb_up and vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals