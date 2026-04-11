#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivots from 1d act as strong support/resistance. Price rejecting these levels with volume shows institutional interest.
# In bull markets, buy at S1/S2; in bear markets, sell at R1/R2. 1d EMA200 filters for trend alignment.
# Low trade frequency expected (~15-25/year) due to strict pivot rejection + volume + trend confluence.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Camarilla pivot levels from previous 1d
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R4 = typical_price + (range_ * 1.1 / 2)
    R3 = typical_price + (range_ * 1.1 / 4)
    R2 = typical_price + (range_ * 1.1 / 6)
    R1 = typical_price + (range_ * 1.1 / 12)
    S1 = typical_price - (range_ * 1.1 / 12)
    S2 = typical_price - (range_ * 1.1 / 6)
    S3 = typical_price - (range_ * 1.1 / 4)
    S4 = typical_price - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4.values)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2.values)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4.values)
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 12h volume average (30-period) for confirmation
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_avg_30[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.8x 30-period average
        vol_confirm = volume[i] > 1.8 * vol_avg_30[i]
        
        # Price rejection conditions (wick rejection of pivot levels)
        # Long: price rejects S1 (long lower wick) OR breaks above S2 with close > S1
        long_reject = (low[i] <= S1_aligned[i] and close[i] > S1_aligned[i] and 
                      (close[i] - low[i]) > (high[i] - low[i]) * 0.6)
        long_break = (close[i] > S2_aligned[i] and close[i] > S1_aligned[i])
        
        # Short: price rejects R1 (long upper wick) OR breaks below R2 with close < R1
        short_reject = (high[i] >= R1_aligned[i] and close[i] < R1_aligned[i] and
                       (high[i] - close[i]) > (high[i] - low[i]) * 0.6)
        short_break = (close[i] < R2_aligned[i] and close[i] < R1_aligned[i])
        
        # 1d EMA200 trend filter
        trend_bullish = close[i] > ema_200_1d_aligned[i]
        trend_bearish = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions
        # Long: S1 rejection/S2 break AND bullish trend AND volume confirmation
        if ((long_reject or long_break) and trend_bullish and vol_confirm and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: R1 rejection/R2 break AND bearish trend AND volume confirmation
        elif ((short_reject or short_break) and trend_bearish and vol_confirm and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Opposite signal (R1 break for long, S1 break for short)
        elif position == 1 and close[i] < S1_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > R1_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals