#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band squeeze breakout with 1-day trend filter and volume confirmation
# Long when price breaks above upper BB during low volatility (BB width < 20th percentile) and 1-day trend up
# Short when price breaks below lower BB during low volatility and 1-day trend down
# Bollinger Band squeeze indicates low volatility primed for breakout; trend filter ensures direction alignment
# Volume confirmation avoids false breakouts; targets 20-50 trades/year for low fee drag

name = "4h_BollingerSqueeze_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2) on 4h
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band width percentile (20-period lookback for regime)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    # Squeeze condition: BB width below 20th percentile (low volatility)
    squeeze = bb_width_percentile < 0.2
    
    # Volume spike: current volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        squeeze_active = squeeze[i]
        vol_spike = volume_spike[i]
        close_price = close[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB during squeeze, 1-day uptrend, volume spike
            if close_price > upper and squeeze_active and ema50_1d_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB during squeeze, 1-day downtrend, volume spike
            elif close_price < lower and squeeze_active and ema50_1d_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB or 1-day trend turns down
            if close_price < sma20[i] or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB or 1-day trend turns up
            if close_price > sma20[i] or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals