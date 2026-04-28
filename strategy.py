#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume confirmation.
# Enter long when price breaks above 4h Camarilla R4 with 1d EMA34 uptrend and volume > 1.5x 20-bar average.
# Enter short when price breaks below 4h Camarilla S4 with 1d EMA34 downtrend and volume confirmation.
# Exit when price retraces to the 4h Camarilla midpoint (R4/S4 average).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla R4/S4 levels provide stronger breakout confirmation than R3/S3, reducing false signals.
# EMA34 on 1d ensures alignment with the primary trend while being responsive enough for 4h entries.
# Volume confirmation filters weak breakouts. This pattern avoids overtrading by using stricter levels.

name = "4h_Camarilla_R4S4_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 5:  # Need at least one complete 4h bar
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels for intraday trading
    # R4 = Close + (High - Low) * 1.1
    # S4 = Close - (High - Low) * 1.1
    # Midpoint = (R4 + S4) / 2 = Close
    camarilla_high = close_4h + (high_4h - low_4h) * 1.1
    camarilla_low = close_4h - (high_4h - low_4h) * 1.1
    camarilla_mid = close_4h  # Midpoint is the close price
    
    # Align Camarilla levels to 4h (shifted by one bar to avoid look-ahead)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (34-period)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        ema_trend_up = close[i] > ema_34_aligned[i]
        ema_trend_down = close[i] < ema_34_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R4, price > EMA34 (uptrend), volume confirm
            if price > camarilla_high_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Camarilla S4, price < EMA34 (downtrend), volume confirm
            elif price < camarilla_low_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals