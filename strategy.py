#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d trend filter
# - Long when price breaks above H3 Camarilla level AND volume > 2.0x 24-bar average AND 1d close > 1d EMA50
# - Short when price breaks below L3 Camarilla level AND volume > 2.0x 24-bar average AND 1d close < 1d EMA50
# - Exit when price returns to pivot point (PP) or opposite extreme is touched
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Camarilla levels provide institutional support/resistance; volume confirms breakout validity; 1d EMA50 filters for trend alignment

name = "12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation: > 2.0x 24-period average
    volume_24_avg = prices['volume'].rolling(window=24, min_periods=24).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_24_avg)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Need at least one full 12h bar to calculate Camarilla levels
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Get the last completed 12h bar for Camarilla calculation
        lookback = i + 1
        if lookback < 1:
            signals[i] = 0.0
            continue
            
        # Use last 12h bar's OHLC to calculate Camarilla pivot levels
        high_12h = prices['high'].iloc[lookback-1]
        low_12h = prices['low'].iloc[lookback-1]
        close_12h = prices['close'].iloc[lookback-1]
        
        # Calculate Camarilla pivot levels for the last completed 12h bar
        range_12h = high_12h - low_12h
        if range_12h <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Camarilla levels
        pp = (high_12h + low_12h + close_12h) / 3.0
        r3 = pp + (range_12h * 1.1 / 4.0)
        s3 = pp - (range_12h * 1.1 / 4.0)
        r4 = pp + (range_12h * 1.1 / 2.0)
        s4 = pp - (range_12h * 1.1 / 2.0)
        
        current_price = prices['close'].iloc[i]
        
        # Skip if any required data is invalid
        if (np.isnan(volume_24_avg[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above R3 with volume spike and 1d uptrend
            if (current_price > r3 and 
                vol_spike.iloc[i] and 
                current_price > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below S3 with volume spike and 1d downtrend
            elif (current_price < s3 and 
                  vol_spike.iloc[i] and 
                  current_price < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to pivot point (PP)
            # 2. Opposite extreme is touched (long exits at S4, short exits at R4)
            if position == 1:
                if current_price <= pp or current_price < s4:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if current_price >= pp or current_price > r4:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals