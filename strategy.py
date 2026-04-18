# 12h_Camarilla_R1S1_Breakout_DailyTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout on 12h with daily EMA trend filter and volume confirmation.
# Buy when price breaks above R1 with volume spike and daily uptrend; short when breaks below S1 with volume spike and daily downtrend.
# Designed for 12-37 trades/year (50-150 total over 4 years) to avoid fee drag while capturing breakout moves in both bull and bear markets.
# Uses daily EMA34 for trend filter to avoid whipsaws in sideways markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    typical = (high + low + close) / 3
    range_val = high - low
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    
    # Camarilla R1/S1 on 12h
    R1, S1 = calculate_camarilla(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values)
    R1_12h = align_htf_to_ltf(prices, df_12h, R1)
    S1_12h = align_htf_to_ltf(prices, df_12h, S1)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_12h[i]) or
            np.isnan(S1_12h[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        R1_val = R1_12h[i]
        S1_val = S1_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and daily uptrend
            if not np.isnan(R1_val) and price > R1_val and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and daily downtrend
            elif not np.isnan(S1_val) and price < S1_val and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below S1 OR daily trend turns down
            if not np.isnan(S1_val) and price < S1_val:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above R1 OR daily trend turns up
            if not np.isnan(R1_val) and price > R1_val:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0