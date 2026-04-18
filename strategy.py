# 1d_RSI50_Cross_With_Volume_Spike
# Hypothesis: RSI crossing 50 indicates momentum shift. Combine with volume spike (>2x 20-period mean) for confirmation.
# Use weekly trend filter: only take long when price > weekly EMA(50), short when price < weekly EMA(50).
# This reduces counter-trend trades. Target low frequency: 10-25 trades/year on 1d.
# Works in bull/bear by following weekly trend.

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
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    period = 14
    if len(close) >= period + 1:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 2x 20-period mean
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    vol_spike = volume > 2 * vol_ma
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    if len(close_1w) >= 50:
        ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period + 1, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        vol_confirm = vol_spike[i]
        
        # Weekly trend filter
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Long: RSI crosses above 50 with volume spike and above weekly EMA
            if rsi[i] > 50 and rsi[i-1] <= 50 and vol_confirm and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50 with volume spike and below weekly EMA
            elif rsi[i] < 50 and rsi[i-1] >= 50 and vol_confirm and below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses below 50
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses above 50
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI50_Cross_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0