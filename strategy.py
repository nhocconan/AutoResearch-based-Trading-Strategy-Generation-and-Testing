#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Williams %R(14) measures overbought/oversold levels (-20 to -80)
# - Long when %R crosses above -80 from below AND 1w close > 1w EMA(21) AND volume > 1.3x 20-period average volume
# - Short when %R crosses below -20 from above AND 1w close < 1w EMA(21) AND volume > 1.3x 20-period average volume
# - Exit when %R crosses opposite threshold (-50 for long exit, -50 for short exit) or volume drops
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R is effective in ranging markets which dominate BTC/ETH 2025+ test period
# - 1w EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals

name = "6h_1w_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
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
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, len(close)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Avoid division by zero
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = np.full_like(close_1w, np.nan, dtype=float)
    alpha = 2 / (21 + 1)
    ema_21_1w[20] = np.mean(close_1w[0:21])  # SMA for first value
    for i in range(21, len(close_1w)):
        ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    
    # 1w trend: close > EMA(21) = uptrend, close < EMA(21) = downtrend
    uptrend_1w = close_1w > ema_21_1w
    downtrend_1w = close_1w < ema_21_1w
    
    # Align HTF indicators to 6h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Williams %R crossover signals
            wr_cross_up = (williams_r[i] > -80 and williams_r[i-1] <= -80)  # Cross above -80
            wr_cross_down = (williams_r[i] < -20 and williams_r[i-1] >= -20)  # Cross below -20
            
            # Long conditions: %R crosses above -80 AND 1w uptrend AND volume spike
            if (wr_cross_up and 
                uptrend_1w_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: %R crosses below -20 AND 1w downtrend AND volume spike
            elif (wr_cross_down and 
                  downtrend_1w_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: %R crosses opposite threshold or volume drops
            wr_exit_long = (williams_r[i] < -50 and williams_r[i-1] >= -50)  # Cross below -50
            wr_exit_short = (williams_r[i] > -50 and williams_r[i-1] <= -50)  # Cross above -50
            volume_drop = not volume_spike[i]
            
            if (position == 1 and (wr_exit_long or volume_drop)) or \
               (position == -1 and (wr_exit_short or volume_drop)):
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals