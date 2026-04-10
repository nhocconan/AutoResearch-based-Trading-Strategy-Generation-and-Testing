#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long when price breaks above H3 Camarilla level AND volume > 1.8x 20-bar average AND 1d close > 1d EMA200
# - Short when price breaks below L3 Camarilla level AND volume > 1.8x 20-bar average AND 1d close < 1d EMA200
# - Exit when price returns to pivot point (PP) or opposite Camarilla level is touched
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Targets ~25-35 trades/year (100-140 total over 4 years) to avoid fee drag
# - Camarilla levels work well in ranging and trending markets, providing institutional support/resistance
# - Volume confirmation ensures breakout validity
# - 1d EMA200 filter ensures alignment with higher timeframe trend

name = "4h_12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 100 or len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Need at least one full 4h bar to calculate Camarilla levels
        if i < 4:
            signals[i] = 0.0
            continue
            
        # Get the last completed 4h bar for Camarilla calculation
        lookback = i + 1
        if lookback < 4:
            signals[i] = 0.0
            continue
            
        # Use last 4h bar's OHLC to calculate Camarilla pivot levels
        high_4h = prices['high'].iloc[lookback-1]
        low_4h = prices['low'].iloc[lookback-1]
        close_4h = prices['close'].iloc[lookback-1]
        
        # Calculate Camarilla pivot levels for the last completed 4h bar
        range_4h = high_4h - low_4h
        if range_4h <= 0:
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
            
        # Camarilla levels
        pp = (high_4h + low_4h + close_4h) / 3.0
        r3 = pp + (range_4h * 1.1 / 4.0)
        s3 = pp - (range_4h * 1.1 / 4.0)
        r4 = pp + (range_4h * 1.1 / 2.0)
        s4 = pp - (range_4h * 1.1 / 2.0)
        
        current_price = prices['close'].iloc[i]
        
        # Skip if any required data is invalid
        if (np.isnan(volume_20_avg[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above R3 with volume spike and 1d uptrend
            if (current_price > r3 and 
                vol_spike.iloc[i] and 
                current_price > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short signal: price breaks below S3 with volume spike and 1d downtrend
            elif (current_price < s3 and 
                  vol_spike.iloc[i] and 
                  current_price < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to pivot point (PP)
            # 2. Opposite extreme is touched (long exits at S4, short exits at R4)
            if position == 1:
                if current_price <= pp or current_price < s4:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30  # Hold long
            elif position == -1:
                if current_price >= pp or current_price > r4:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30  # Hold short
    
    return signals