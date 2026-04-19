#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 12h trend bullish (price > 12h EMA34) AND volume > 1.3x 12h average volume
# Short when Williams %R > -20 (overbought) AND 12h trend bearish (price < 12h EMA34) AND volume > 1.3x 12h average volume
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# Williams %R identifies short-term extremes, 12h EMA34 provides trend filter, volume confirms conviction.
# Target: 15-35 trades/year per symbol.
name = "6h_WilliamsR_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    
    # Williams %R (14 periods) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 12h average volume (20-period) for confirmation
    vol_ma_12h = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 34, 20)  # Ensure Williams %R, EMA34, and vol MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema34 = ema34_12h_aligned[i]
        vol_ma = vol_ma_12h_aligned[i]
        vol = volume[i]
        
        # Trend and volume conditions
        bullish_trend = price > ema34
        bearish_trend = price < ema34
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long entry: oversold + bullish trend + volume confirmation
            if wr < -80 and bullish_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: overbought + bearish trend + volume confirmation
            elif wr > -20 and bearish_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (exit oversold territory)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (exit overbought territory)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals