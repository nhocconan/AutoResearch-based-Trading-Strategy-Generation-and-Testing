#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band reversal with 1w trend filter and volume confirmation
# Long when: price touches lower BB(20,2) AND closes back inside + 1w EMA50 uptrend + volume spike
# Short when: price touches upper BB(20,2) AND closes back inside + 1w EMA50 downtrend + volume spike
# Bollinger Bands capture mean reversion extremes, 1w EMA50 filters for higher timeframe trend
# Volume confirmation ensures institutional participation in reversals
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years)

name = "1d_BollingerReversal_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need BB and EMA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long: price touched lower band today AND closed back inside + uptrend + volume
            touched_lower = low[i] <= lower
            closed_inside = close[i] > lower
            if touched_lower and closed_inside and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price touched upper band today AND closed back inside + downtrend + volume
            touched_upper = high[i] >= upper
            closed_inside_short = close[i] < upper
            if touched_upper and closed_inside_short and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price touches upper band OR trend changes
            if high[i] >= upper_band[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price touches lower band OR trend changes
            if low[i] <= lower_band[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals