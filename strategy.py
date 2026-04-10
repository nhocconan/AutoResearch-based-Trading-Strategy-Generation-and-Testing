#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND price > 50-week EMA (bullish trend) AND volume > 1.5x 20-day average volume
# - Short when Williams %R(14) > -20 (overbought) AND price < 50-week EMA (bearish trend) AND volume > 1.5x 20-day average volume
# - Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Williams %R identifies overextended moves likely to revert
# - Weekly EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals

name = "1d_1w_williamsr_meanreversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Williams %R (14-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    highest_high = rolling_max(high, 14)
    lowest_low = rolling_min(low, 14)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 1d volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w 50-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.zeros_like(close_1w)
    ema_50_1w[0] = close_1w[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_1w)):
        ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Align HTF indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND price above weekly EMA AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND price below weekly EMA AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
            exit_long = (position == 1 and williams_r[i] > -50)
            exit_short = (position == -1 and williams_r[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals