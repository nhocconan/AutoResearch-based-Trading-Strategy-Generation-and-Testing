#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Bollinger Band squeeze with daily trend filter and volume confirmation
# Long when BB width at 20-period low, price breaks above upper band, daily trend up, volume spike
# Short when BB width at 20-period low, price breaks below lower band, daily trend down, volume spike
# Bollinger squeeze indicates low volatility primed for breakout; volume confirms institutional interest
# Daily trend filter ensures alignment with higher timeframe momentum
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_BollingerSqueeze_DailyTrend_Volume"
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
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + (bb_std * std_bb)
    lower_bb = sma_bb - (bb_std * std_bb)
    bb_width = upper_bb - lower_bb
    
    # BB width percentile for squeeze detection (lowest 20% = squeeze)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(sma_bb[i]) or 
            np.isnan(std_bb[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        bb_width_pct = bb_width_percentile[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: BB squeeze (width in lowest 20%), price breaks above upper band, daily uptrend, volume spike
            if bb_width_pct <= 0.2 and close[i] > upper_bb[i] and ema50_1d_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: BB squeeze (width in lowest 20%), price breaks below lower band, daily downtrend, volume spike
            elif bb_width_pct <= 0.2 and close[i] < lower_bb[i] and ema50_1d_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band or daily trend turns down
            if close[i] < sma_bb[i] or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band or daily trend turns up
            if close[i] > sma_bb[i] or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals