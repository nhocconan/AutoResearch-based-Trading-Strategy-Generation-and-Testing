#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) AND 1w close > 1w EMA(21) AND volume > 1.5x 12h average volume
# - Short when Williams %R(14) crosses below -20 (overbought) AND 1w close < 1w EMA(21) AND volume > 1.5x 12h average volume
# - Exit when Williams %R returns to neutral zone (-50) or opposite extreme is reached
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Williams %R identifies overextended moves ripe for mean reversion
# - 1w EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals

name = "12h_1w_williamsr_meanreversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Williams %R (14-period)
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
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = np.zeros_like(close_1w)
    ema_21_1w[0] = close_1w[0]
    alpha = 2 / (21 + 1)
    for i in range(1, len(close_1w)):
        ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    
    # Align HTF indicators to 12h timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R crosses above -80 AND 1w trend up AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_21_1w_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R crosses below -20 AND 1w trend down AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_21_1w_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R returns to neutral (-50) or reaches opposite extreme
            exit_long = (position == 1 and (williams_r[i] >= -50 or williams_r[i] <= -20))
            exit_short = (position == -1 and (williams_r[i] <= -50 or williams_r[i] >= -80))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals