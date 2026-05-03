#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# Donchian breakouts capture strong momentum moves. The 12h EMA50 filter ensures we only trade
# in the direction of the higher timeframe trend, reducing whipsaws. Volume spike confirms
# institutional participation. This combination has proven effective on SOLUSDT and should
# generalize to BTC/ETH by focusing on strong trending moves with confirmation.

name = "4h_Donchian20_12hEMA50_VolumeSpike_Trend"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 4h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        highest = highest_high[i]
        lowest = lowest_low[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(highest) or np.isnan(lowest) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine breakout conditions
        long_breakout = close_val > highest
        short_breakout = close_val < lowest
        
        # Determine regime: bull if close > 12h EMA50, bear if close < 12h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Generate signals
        if position == 0:
            # Long: bullish breakout in bull regime with volume spike
            if long_breakout and is_bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout in bear regime with volume spike
            elif short_breakout and is_bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on bearish breakout or regime change to bear
            if short_breakout or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on bullish breakout or regime change to bull
            if long_breakout or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals