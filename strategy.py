#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA(34) trend filter and volume confirmation
# Long when price breaks above Camarilla R3, price > 4h EMA34, and volume > 2.0x 24-bar average
# Short when price breaks below Camarilla S3, price < 4h EMA34, and volume > 2.0x 24-bar average
# Uses 4h EMA for higher timeframe trend alignment (matches experiment HTF)
# Volume spike confirms breakout strength
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Discrete position sizing (0.20) to minimize fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Works in bull (breakouts above rising EMA) and bear (breakdowns below falling EMA)

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_Volume_v1"
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
    
    # Get 4h data for EMA(34) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 4h for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA to 1h timeframe (wait for completed 4h bar)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla pivots on 1h (using previous bar's high-low-close)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    hl_range = pd.Series(high).shift(1) - pd.Series(low).shift(1)
    camarilla_r3 = pd.Series(close).shift(1) + (hl_range * 1.1 / 4)
    camarilla_s3 = pd.Series(close).shift(1) - (hl_range * 1.1 / 4)
    
    # Volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 24) + 1  # EMA(34) + volume MA(24) + shift(1) warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r3.iloc[i]) or 
            np.isnan(camarilla_s3.iloc[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R3, price > 4h EMA34, volume spike
            if (close[i] > camarilla_r3.iloc[i] and 
                close[i] > ema_34_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price < Camarilla S3, price < 4h EMA34, volume spike
            elif (close[i] < camarilla_s3.iloc[i] and 
                  close[i] < ema_34_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S3 or price < 4h EMA34
            if (close[i] < camarilla_s3.iloc[i] or 
                close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 or price > 4h EMA34
            if (close[i] > camarilla_r3.iloc[i] or 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals