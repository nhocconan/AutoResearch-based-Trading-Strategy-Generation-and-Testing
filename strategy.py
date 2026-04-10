#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h volume confirmation and 1d trend filter
# - Bollinger Bands (20,2) squeeze identifies low volatility periods primed for breakout
# - Long when BB width < 20th percentile AND close breaks above upper band AND 12h volume > 1.5x average AND 1d close > 1d EMA50
# - Short when BB width < 20th percentile AND close breaks below lower band AND 12h volume > 1.5x average AND 1d close < 1d EMA50
# - Exit when price returns to middle band (20 SMA) or opposite band is touched
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Bollinger squeeze works well in ranging markets (2022-2025) and catches impulsive moves
# - Volume confirmation ensures breakout has participation
# - 1d trend filter avoids trading against higher timeframe momentum

name = "6h_12h_1d_bb_squeeze_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Bollinger Bands (20,2) on 6h data
    bb_period = 20
    bb_std = 2
    close_6h = prices['close'].values
    
    # Calculate middle band (SMA)
    sma_20 = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).mean().values
    
    # Calculate standard deviation
    std_20 = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).std().values
    
    # Calculate upper and lower bands
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    
    # Calculate Bollinger Band width (normalized by middle band)
    bb_width = np.where(sma_20 != 0, (upper_bb - lower_bb) / sma_20, 0)
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window to avoid look-ahead)
    bb_width_percentile = np.zeros(n)
    for i in range(n):
        if i >= bb_period:
            # Use historical data up to i for percentile calculation
            historical_width = bb_width[:i+1]
            valid_width = historical_width[~np.isnan(historical_width)]
            if len(valid_width) >= bb_period:
                bb_width_percentile[i] = np.percentile(valid_width, 20)
            else:
                bb_width_percentile[i] = np.inf  # No squeeze if insufficient data
        else:
            bb_width_percentile[i] = np.inf
    
    # Squeeze condition: BB width below 20th percentile
    squeeze_condition = bb_width < bb_width_percentile
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: squeeze breakout above upper band with volume spike and 1d uptrend
            if (squeeze_condition[i] and 
                close_6h[i] > upper_bb[i] and 
                vol_spike.iloc[i] and 
                close_6h[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: squeeze breakout below lower band with volume spike and 1d downtrend
            elif (squeeze_condition[i] and 
                  close_6h[i] < lower_bb[i] and 
                  vol_spike.iloc[i] and 
                  close_6h[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to middle band (20 SMA)
            # 2. Price touches opposite band (contrarian exit)
            if position == 1:
                if close_6h[i] <= sma_20[i] or close_6h[i] >= upper_bb[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if close_6h[i] >= sma_20[i] or close_6h[i] <= lower_bb[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals