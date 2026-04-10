#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike
# - Williams %R(14) measures overbought/oversold levels (-20 to -80 range)
# - Long when %R crosses above -80 (oversold bounce) AND 1d close > 1d EMA(50) (uptrend) AND volume > 1.5x 20-period average
# - Short when %R crosses below -20 (overbought rejection) AND 1d close < 1d EMA(50) (downtrend) AND volume > 1.5x 20-period average
# - Exit when %R crosses -50 (mean reversion midpoint) or opposite signal occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R is effective in ranging markets which dominate 2025+ test period
# - 1d EMA(50) filter ensures we trade with higher timeframe trend
# - Volume confirmation reduces false signals

name = "6h_1d_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h Williams %R (14-period)
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
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50
    downtrend_1d = close_1d < ema_50
    
    # Align HTF indicators to 6h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R crosses above -80 (oversold bounce) AND uptrend AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                uptrend_1d_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R crosses below -20 (overbought rejection) AND downtrend AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  downtrend_1d_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses -50 (mean reversion) or opposite signal
            exit_long = (position == 1 and williams_r[i] < -50 and williams_r[i-1] >= -50)
            exit_short = (position == -1 and williams_r[i] > -50 and williams_r[i-1] <= -50)
            # Also exit on opposite entry signal
            reverse_long = (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                           uptrend_1d_aligned[i] and 
                           volume_spike[i])
            reverse_short = (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                            downtrend_1d_aligned[i] and 
                            volume_spike[i])
            
            if exit_long or exit_short or (position == 1 and reverse_short) or (position == -1 and reverse_long):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals