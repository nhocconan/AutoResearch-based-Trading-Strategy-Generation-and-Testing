#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week RSI mean reversion with 1-day volume confirmation.
# Long when weekly RSI < 30 (oversold) AND daily volume > 1.5x 20-day average.
# Short when weekly RSI > 70 (overbought) AND daily volume > 1.5x 20-day average.
# Exit when weekly RSI crosses back above 50 (for long) or below 50 (for short).
# Uses weekly RSI for extreme conditions and daily volume to confirm institutional interest.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "12h_WklyRSI_1dVol_MeanRev"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on weekly data
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = 50  # Initialize first values
    
    # Align weekly RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Sufficient warmup for RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(rsi_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: weekly RSI < 30 (oversold) AND volume spike
            long_cond = (rsi_aligned[i] < 30) and volume_filter[i]
            # Short conditions: weekly RSI > 70 (overbought) AND volume spike
            short_cond = (rsi_aligned[i] > 70) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly RSI crosses back above 50
            if rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly RSI crosses back below 50
            if rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals