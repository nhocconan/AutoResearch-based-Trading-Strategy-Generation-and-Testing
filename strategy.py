#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot breakout with volume confirmation
# - Long: price breaks above weekly Camarilla H3 level, volume > 1.4x 20-period avg, price > 1d EMA(50) (trend filter)
# - Short: price breaks below weekly Camarilla L3 level, volume > 1.4x 20-period avg, price < 1d EMA(50) (trend filter)
# - Exit: price returns to weekly Camarilla pivot point (PP)
# - Uses 1d EMA(50) for trend bias to avoid counter-trend trades
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# - Weekly Camarilla levels provide stronger support/resistance than daily, reducing false breakouts
# - Volume confirmation ensures breakouts have conviction
# - Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend

name = "6h_1w_camarilla_volume_trend_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return signals
    
    # Pre-compute 1w OHLC for Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1w bar
    # PP = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # We'll use H3 (R3) and L3 (S3) for breakouts, PP for exit
    PP = (high_1w + low_1w + close_1w) / 3
    R3 = close_1w + ((high_1w - low_1w) * 1.1 / 4)
    S3 = close_1w - ((high_1w - low_1w) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        PP_level = PP_aligned[i]
        R3_level = R3_aligned[i]
        S3_level = S3_aligned[i]
        
        # Volume confirmation: current volume > 1.4x 20-period average
        vol_confirm = volume_current > 1.4 * volume_sma_20[i]
        
        # Trend bias from 1d EMA(50)
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above weekly Camarilla R3, volume confirmation, long bias
        if close_price > R3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below weekly Camarilla L3, volume confirmation, short bias
        if close_price < S3_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to weekly pivot point
            exit_long = close_price <= PP_level
        elif position == -1:
            # Exit short if price returns to weekly pivot point
            exit_short = close_price >= PP_level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals