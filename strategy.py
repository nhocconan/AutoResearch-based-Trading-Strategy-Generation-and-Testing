#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d EMA200 trend filter and volume spike confirmation
# Long when Williams %R < -80 (oversold) AND close > 1d EMA200 AND volume > 2.0x 20-bar avg
# Short when Williams %R > -20 (overbought) AND close < 1d EMA200 AND volume > 2.0x 20-bar avg
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses discrete position sizing (0.25) to balance return and fee drag. Target: 15-30 trades/year on 4h.
# Works in bull markets via mean reversion from oversold levels in uptrend, works in bear via 
# overbought bounces in downtrend. Volume spike requirement filters weak signals and captures 
# institutional participation. 4h timeframe balances responsiveness with low trade frequency.

name = "4h_WilliamsR_Extreme_1dEMA200_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    # Calculate EMA(200) on 1d close
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams %R on 4h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for low trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 14)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_200_1d_aligned[i]
        wr = williams_r[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND close > 1d EMA200 AND volume confirmation
            if wr < -80 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND close < 1d EMA200 AND volume confirmation
            elif wr > -20 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R crosses above -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R crosses below -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals