#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses daily EMA34 for robust trend alignment to avoid counter-trend trades
# Camarilla R3/S3 levels provide precise intraday support/resistance for breakouts
# Volume spike (2.0x 20-period average) ensures participation and reduces false breakouts
# Discrete position sizing 0.25 balances exposure and minimizes fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    # S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # We use daily OHLC to calculate levels for the 4h bars within that day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Group by date to get daily OHLC
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    for date in unique_dates:
        mask = (dates == date)
        if not np.any(mask):
            continue
        # Get first 4h bar of the day for daily OHLC (assuming 4h data aligned to UTC)
        day_open = prices.loc[mask, 'open'].iloc[0]
        day_high = prices.loc[mask, 'high'].max()
        day_low = prices.loc[mask, 'low'].min()
        day_close = prices.loc[mask, 'close'].iloc[-1]
        
        # Calculate Camarilla levels for this day
        range_hl = day_high - day_low
        camarilla_r3_val = day_close + (range_hl * 1.1 / 4)
        camarilla_s3_val = day_close - (range_hl * 1.1 / 4)
        
        camarilla_r3[mask] = camarilla_r3_val
        camarilla_s3[mask] = camarilla_s3_val
    
    # Calculate 4h volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and Camarilla levels)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 AND above 1d EMA34 AND volume confirm
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND below 1d EMA34 AND volume confirm
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR below 1d EMA34
            if (close[i] < camarilla_s3[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR above 1d EMA34
            if (close[i] > camarilla_r3[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals