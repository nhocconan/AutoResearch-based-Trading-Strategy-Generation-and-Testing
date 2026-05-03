#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA(34) trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) + price > 1d EMA(34) + volume spike
# Short when Williams %R > -20 (overbought) + price < 1d EMA(34) + volume spike
# Uses 1d EMA(34) for trend alignment to reduce whipsaw in ranging markets
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Williams %R calculated from prior 6h session to avoid look-ahead
# Designed for low trade frequency (12-37/year on 6h) to minimize fee drag
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)

name = "6h_WilliamsR_Extremes_1dEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R from prior 6h session (using 6h data)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=14, min_periods=14).max().shift(1).values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().shift(1).values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 14, 20)  # 1d EMA(34), Williams %R(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + price > 1d EMA(34) + volume spike
            if (williams_r[i] < -80 and close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) + price < 1d EMA(34) + volume spike
            elif (williams_r[i] > -20 and close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or price < 1d EMA(34)
            if (williams_r[i] > -20 or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or price > 1d EMA(34)
            if (williams_r[i] < -80 or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals