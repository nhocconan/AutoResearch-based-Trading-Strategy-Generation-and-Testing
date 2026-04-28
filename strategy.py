#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA200 trend filter and volume confirmation.
# Enter long when price breaks above 1h Camarilla R3 with 4h EMA200 uptrend and volume > 2.0x 20-bar average.
# Enter short when price breaks below 1h Camarilla S3 with 4h EMA200 downtrend and volume confirmation.
# Exit when price retraces to the 1h Camarilla midpoint (R3/S3 average).
# Uses discrete position sizing (0.20) to limit drawdown and reduce fee churn.
# Target: 60-150 total trades over 4 years (15-37/year).
# Camarilla levels provide precise intraday support/resistance. EMA200 on 4h ensures trend alignment.
# Volume confirmation filters weak breakouts. Session filter (08-20 UTC) reduces noise trades.
# Designed to work in both bull and bear markets via trend filter and mean-reversion exit.

name = "1h_Camarilla_R3S3_Breakout_4hEMA200_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data for Camarilla calculation (based on previous 1h bar)
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_1h) < 5:  # Need at least one complete 1h bar
        return np.zeros(n)
    
    # Calculate 1h Camarilla levels (based on previous 1h bar)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla levels for intraday trading
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    # Midpoint = (R3 + S3) / 2 = Close
    camarilla_high = close_1h + (high_1h - low_1h) * 1.1 / 2.0
    camarilla_low = close_1h - (high_1h - low_1h) * 1.1 / 2.0
    camarilla_mid = close_1h  # Midpoint is the close price
    
    # Align Camarilla levels to 1h (shifted by one bar to avoid look-ahead)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1h, camarilla_low)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1h, camarilla_mid)
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 200:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 4h EMA (200-period)
    close_4h = df_4h['close'].values
    ema_200 = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA to 1h
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 4h EMA200 trend filter: price > EMA200 = uptrend, price < EMA200 = downtrend
        ema_trend_up = close[i] > ema_200_aligned[i]
        ema_trend_down = close[i] < ema_200_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R3, price > EMA200 (uptrend), volume confirm
            if price > camarilla_high_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: price < Camarilla S3, price < EMA200 (downtrend), volume confirm
            elif price < camarilla_low_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals