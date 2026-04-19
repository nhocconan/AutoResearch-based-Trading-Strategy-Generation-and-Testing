#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 14-period Williams %R with 1-week trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold), price > 1w EMA50, volume > 1.5x 20-period average.
# Short when Williams %R crosses below -20 (overbought), price < 1w EMA50, volume > 1.5x 20-period average.
# Uses discrete position sizes (0.25) to minimize churn. Designed for 12h timeframe
# to capture mean reversion in both bull and bear markets with trend filter.
# Target: 25-40 trades/year per symbol (~100-160 total over 4 years).
name = "12h_WilliamsR_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams %R calculation (14-period)
    williams_r = np.full_like(close, np.nan)
    for i in range(14, len(high)):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Align 1w EMA50 to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure EMA50 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        williams_r_val = williams_r[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 (from below), price above 1w EMA50, volume confirmation
            if i > start_idx and williams_r[i-1] <= -80 and williams_r_val > -80 and price > ema_50_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 (from above), price below 1w EMA50, volume confirmation
            elif i > start_idx and williams_r[i-1] >= -20 and williams_r_val < -20 and price < ema_50_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Williams %R crosses below -50 (momentum loss) or price crosses below 1w EMA50
            if i > start_idx and (williams_r[i-1] > -50 and williams_r_val <= -50 or price < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Williams %R crosses above -50 (momentum loss) or price crosses above 1w EMA50
            if i > start_idx and (williams_r[i-1] < -50 and williams_r_val >= -50 or price > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals