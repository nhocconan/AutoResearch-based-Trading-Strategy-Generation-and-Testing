#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA(50) trend filter with volume confirmation
# Williams %R(14) identifies overbought/oversold conditions (above -20 = overbought, below -80 = oversold)
# Only take long signals when %R crosses above -80 from below (oversold bounce) in uptrend (price > EMA)
# Only take short signals when %R crosses below -20 from above (overbought rejection) in downtrend (price < EMA)
# Volume filter requires current volume > 1.3x 20-period average to confirm momentum
# Targets 15-35 trades/year (60-140 over 4 years) to minimize fee drag while capturing mean reversion within trend
# Works in bull/bear by only trading with higher timeframe trend direction

name = "6h_williamsr_1d_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[13] = np.mean(tr[:14])  # Fixed indexing: at index 13 we have 14 values (0-13)
            for i in range(14, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 14-period Williams %R
    williams_r = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(13, n):  # Start at index 13 for 14-period lookback (0-13)
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # 50-period EMA on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(williams_r[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Williams %R conditions
        wr_oversold = williams_r[i] < -80
        wr_overbought = williams_r[i] > -20
        wr_cross_up_oversold = williams_r[i] > -80 and williams_r[i-1] <= -80
        wr_cross_down_overbought = williams_r[i] < -20 and williams_r[i-1] >= -20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R becomes overbought or stoploss hit
            if (wr_overbought or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R becomes oversold or stoploss hit
            if (wr_oversold or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: Williams %R crosses above -80 from oversold with volume and above EMA (bullish)
            if (wr_cross_up_oversold and volume_filter and 
                close[i] > ema_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R crosses below -20 from overbought with volume and below EMA (bearish)
            elif (wr_cross_down_overbought and volume_filter and 
                  close[i] < ema_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals