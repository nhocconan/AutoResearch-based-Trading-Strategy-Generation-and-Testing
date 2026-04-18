# 6h_Pivot_R1_S1_Breakout_HTF_Trend_Volume
# Hypothesis: On 6h timeframe, price respects daily (1d) Camarilla pivot levels R1/S1.
# In strong 12h trends (price > EMA34), breakouts of R1/S1 with volume confirmation
# (volume > 1.5x 20-period mean) continue the trend. In weak trends (price < EMA34),
# reversals at R1/S1 with volume capture mean reversion. This adapts to trend regime
# using a single HTF filter (12h EMA34) to avoid overtrading. Targets 15-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels R1 and S1 (based on previous day)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            R1[i] = prev_close + 1.1 * range_ / 12
            S1[i] = prev_close - 1.1 * range_ / 12
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    if len(close_12h) >= 34:
        ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    else:
        ema_12h = np.full_like(close_12h, np.nan)
    
    # Align all data to 6h timeframe
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    ema_12h_6h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 34, 20) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(ema_12h_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 12h EMA34
        uptrend = close[i] > ema_12h_6h[i]
        downtrend = close[i] < ema_12h_6h[i]
        
        if position == 0:
            # In uptrend: breakout above R1 with volume = long
            # In downtrend: breakdown below S1 with volume = short
            if uptrend and close[i] > R1_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            elif downtrend and close[i] < S1_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < S1_6h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal signal)
            if close[i] > R1_6h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_HTF_Trend_Volume"
timeframe = "6h"
leverage = 1.0