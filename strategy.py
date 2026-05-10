#3/149218
# 1d_1w_Trend_Follow_Volume
# Hypothesis: Use 1w EMA trend filter on 1d timeframe with volume confirmation to capture major trends while avoiding whipsaws.
# Weekly trend provides strong directional bias, daily entries with volume filter reduce false signals.
# Designed for low frequency (7-25 trades/year) to minimize fee drag and work in both bull and bear markets.

name = "1d_1w_Trend_Follow_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA and volume calculations
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1d EMA34 for entry filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1w EMA34 for trend filter (needs confirmation from previous week)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_shifted = np.roll(ema_34_1w, 1)  # Use previous week's EMA to avoid look-ahead
    ema_34_1w_shifted[0] = np.nan
    
    # 1d volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align all indicators to daily timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_shifted)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or \
           np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above both EMAs and strong volume
            if close[i] > ema_34_1d_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below both EMAs and strong volume
            elif close[i] < ema_34_1d_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below either EMA
            if close[i] < ema_34_1d_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above either EMA
            if close[i] > ema_34_1d_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals