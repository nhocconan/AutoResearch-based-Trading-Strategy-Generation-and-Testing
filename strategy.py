#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: Daily Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivots identify institutional support/resistance. Weekly trend filters direction.
# Volume > 1.5x 20-day average confirms institutional participation. Designed for low trade frequency (~10-25/year)
# to minimize fee flood. Works in bull markets via long bounces at support and bear markets via short rejections at resistance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Calculate Camarilla pivot levels for today using yesterday's OHLC
        if i == 0:
            signals[i] = 0.0
            continue
            
        # Previous day's OHLC
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla levels
        range_ = ph - pl
        if range_ <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Resistance levels
        r4 = pc + range_ * 1.1 / 2
        r3 = pc + range_ * 1.1/4
        r2 = pc + range_ * 1.1/6
        r1 = pc + range_ * 1.1/12
        
        # Support levels
        s1 = pc - range_ * 1.1/12
        s2 = pc - range_ * 1.1/6
        s3 = pc - range_ * 1.1/4
        s4 = pc - range_ * 1.1/2
        
        # Trend filter: price above/below weekly EMA200
        trend_bullish = close[i] > ema_200_1w_aligned[i]
        trend_bearish = close[i] < ema_200_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        # Long: Price touches/slightly penetrates S3/S4 in uptrend with volume
        if trend_bullish and vol_confirm:
            if low[i] <= s3 * 1.005 and close[i] > s3:  # Touch S3 and close back above
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            elif low[i] <= s4 * 1.005 and close[i] > s4:  # Touch S4 and close back above
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25 if position == 1 else 0.0
        # Short: Price touches/slightly penetrates R3/R4 in downtrend with volume
        elif trend_bearish and vol_confirm:
            if high[i] >= r3 * 0.995 and close[i] < r3:  # Touch R3 and close back below
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            elif high[i] >= r4 * 0.995 and close[i] < r4:  # Touch R4 and close back below
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25 if position == -1 else 0.0
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals