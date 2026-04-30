#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h trend filter and volume confirmation
# Bollinger Band squeeze (BB width < 20th percentile) indicates low volatility, primed for breakout
# Breakout direction determined by 12h EMA50 trend (above/below)
# Volume spike (2.0x 20-period average) confirms breakout validity
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull/bear markets: squeeze breakouts capture volatility expansion regardless of direction.

name = "6h_BBandSqueeze_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = upper_band - lower_band
    
    # Calculate BB width percentile (20-period lookback for squeeze detection)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    squeeze_condition = bb_width_percentile < 0.20  # BB width < 20th percentile = squeeze
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)  # warmup for BBands and 12h EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_sma_20 = sma_20[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_squeeze = squeeze_condition[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require squeeze breakout with volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above upper band AND 12h trend is up (price > EMA50)
                if curr_close > curr_upper and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower band AND 12h trend is down (price < EMA50)
                elif curr_close < curr_lower and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price returns to middle band (mean reversion) or squeeze breaks down
            if curr_close <= curr_sma_20 or not curr_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to middle band or squeeze breaks down
            if curr_close >= curr_sma_20 or not curr_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals