#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R extreme with 4h EMA200 trend filter and volume confirmation
# Long when Williams %R(14) < -80 (oversold) AND close > 4h EMA200 AND volume > 2.0x 20-bar avg
# Short when Williams %R(14) > -20 (overbought) AND close < 4h EMA200 AND volume > 2.0x 20-bar avg
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme reached
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-30 trades/year on 1h.
# Works in bull markets via mean reversion from oversold, works in bear via selling into overbought
# during rallies. Volume confirmation filters weak breakouts and ensures participation.

name = "1h_WilliamsR_Extreme_4hEMA200_Trend_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA200
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    # Calculate EMA(200) on 4h close
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMA200 to 1h timeframe
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Williams %R(14) calculation
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_200_4h_aligned[i]
        wr = williams_r[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND close > 4h EMA200 AND volume confirmation
            if wr < -80 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when Williams %R > -20 (overbought) AND close < 4h EMA200 AND volume confirmation
            elif wr > -20 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R returns to -50 or reaches -20
            if wr >= -50:  # Mean reversion exit
                signals[i] = 0.0
                position = 0
            elif wr > -20:  # If it reaches overbought, exit to avoid giving back profits
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when Williams %R returns to -50 or reaches -80
            if wr <= -50:  # Mean reversion exit
                signals[i] = 0.0
                position = 0
            elif wr < -80:  # If it reaches oversold, exit to avoid giving back profits
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals