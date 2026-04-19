#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) and weekly trend is bullish (price > weekly EMA50) and volume > 1.5x average.
# Short when Williams %R > -20 (overbought) and weekly trend is bearish (price < weekly EMA50) and volume > 1.5x average.
# Exit when Williams %R crosses -50 (mean reversion complete) or when weekly trend changes.
# Uses Williams %R for overextended conditions, weekly EMA for trend filter, volume for confirmation.
# Target: 15-30 trades/year per symbol to stay within frequency limits.
name = "6h_WilliamsR_MeanReversion_WeeklyTrend_Volume"
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
    
    # Calculate Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 6h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure Williams %R and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r[i]
        weekly_ema = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80), weekly trend bullish (price > weekly EMA), volume confirmation
            if wr < -80 and price > weekly_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20), weekly trend bearish (price < weekly EMA), volume confirmation
            elif wr > -20 and price < weekly_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion) or weekly trend turns bearish
            if wr > -50 or price < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion) or weekly trend turns bullish
            if wr < -50 or price > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals