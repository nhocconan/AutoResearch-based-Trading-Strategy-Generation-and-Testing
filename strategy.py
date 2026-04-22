#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1-day EMA trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions on 12h chart.
# Trend filter uses daily EMA34 to ensure trades align with higher timeframe trend.
# Volume confirmation requires current volume > 1.5x 20-period average.
# Designed to work in both bull and bear markets by only taking trades in direction of daily trend.
# Targets 15-25 trades/year with disciplined risk control.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 12h data (requires high, low, close)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        wr = williams_r[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: oversold Williams %R (< -80) + price above daily EMA + volume spike
            if wr < -80 and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought Williams %R (> -20) + price below daily EMA + volume spike
            elif wr > -20 and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R crosses back through -50 (middle) or opposing extreme
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R rises above -50 (overbought territory) or reaches overbought extreme
                if wr > -50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R falls below -50 (oversold territory) or reaches oversold extreme
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_EMA34_Volume"
timeframe = "12h"
leverage = 1.0