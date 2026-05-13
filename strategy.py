#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation. 
# Long when price breaks above R3 with volume > 1.5x MA20 and close > 1d EMA50.
# Short when price breaks below S3 with volume > 1.5x MA20 and close < 1d EMA50.
# Uses discrete position sizing (0.25) to limit trades to target range (12-37/year) and minimize fee drag.
# Designed to work in bull markets via breakout continuation and in bear markets via breakdown continuation.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # We use the previous completed 12h bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma20)
    
    # Discrete position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or \
           np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and price > 1d EMA50
            if close[i] > camarilla_r3[i] and volume_confirm[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = position_size
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and price < 1d EMA50
            elif close[i] < camarilla_s3[i] and volume_confirm[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -position_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reverse breakdown) OR close < 1d EMA50 (trend break)
            if close[i] < camarilla_s3[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reverse breakout) OR close > 1d EMA50 (trend break)
            if close[i] > camarilla_r3[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals