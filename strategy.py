#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 1h primary timeframe for signal generation with Camarilla pivot breakouts
# 4h EMA50 trend filter provides higher timeframe bias (price > EMA50 for longs, < for shorts)
# Volume confirmation (1.8x 20-period average) filters for strong participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete position sizing (0.20) balances profit potential with fee drag minimization
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels provide objective support/resistance, reducing false signals
# Works in both bull and bear markets by only trading in direction of 4h trend

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels from previous day (using typical price)
    typical_price = (high + low + close) / 3
    typical_series = pd.Series(typical_price)
    typical_ma = typical_series.rolling(window=24, min_periods=24).mean().shift(1).values  # 24*1h = 1 day
    typical_std = typical_series.rolling(window=24, min_periods=24).std().shift(1).values
    camarilla_h3 = typical_ma + 1.1 * typical_std * 1.5  # R3 level
    camarilla_l3 = typical_ma - 1.1 * typical_std * 1.5  # S3 level
    camarilla_h4 = typical_ma + 1.1 * typical_std * 2.0  # R4 level (stop)
    camarilla_l4 = typical_ma - 1.1 * typical_std * 2.0  # S4 level (stop)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and Camarilla calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla H3 + volume spike + price > 4h EMA50
            if close[i] > camarilla_h3[i] and volume_spike[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla L3 + volume spike + price < 4h EMA50
            elif close[i] < camarilla_l3[i] and volume_spike[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Camarilla L3 or price < 4h EMA50
            if close[i] < camarilla_l3[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close above Camarilla H3 or price > 4h EMA50
            if close[i] > camarilla_h3[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals