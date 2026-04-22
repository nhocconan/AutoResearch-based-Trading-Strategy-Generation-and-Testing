#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4-hour Williams %R mean-reversion with 12h trend filter and volume confirmation
    # Williams %R identifies overbought/oversold conditions at key levels
    # 12h EMA50 provides trend direction to avoid counter-trend trades
    # Volume spike confirms institutional participation in reversals
    # Works in both bull/bear markets: mean reversion in ranges, trend-following in trends
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Williams %R on 4h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Handle division by zero
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold (-80 to -100) with volume spike and price above 12h EMA50 (uptrend bias)
            if williams_r[i] <= -80 and vol_spike[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (-20 to 0) with volume spike and price below 12h EMA50 (downtrend bias)
            elif williams_r[i] >= -20 and vol_spike[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to neutral zone (-50) or opposite extreme
            if position == 1:
                if williams_r[i] >= -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r[i] <= -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Williams_R_MeanReversion_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0