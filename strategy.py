#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R oversold (< -80) and 12h EMA50 trending up with volume spike.
# Short when Williams %R overbought (> -20) and 12h EMA50 trending down with volume spike.
# Williams %R identifies mean reversion points in ranging markets, filtered by 12h trend to avoid counter-trend trades.
# Volume spike confirms momentum at reversal points.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).
name = "6h_WilliamsR_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 50)  # Williams %R needs 14, volume needs 20, EMA needs 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: Williams %R oversold AND 12h EMA trending up AND volume confirmation
            if wr < -80 and close[i] > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought AND 12h EMA trending down AND volume confirmation
            elif wr > -20 and close[i] < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R becomes overbought or trend turns down
            if wr > -20 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R becomes oversold or trend turns up
            if wr < -80 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals