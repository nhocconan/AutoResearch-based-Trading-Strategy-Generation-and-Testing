#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation (>1.5x 20-period average)
# Camarilla pivot points identify key intraday support/resistance levels; breakouts above R3 or below S3 with volume
# indicate strong institutional participation. 4h EMA50 ensures alignment with higher timeframe trend to avoid
# counter-trend whipsaws. Volume confirmation filters breakouts lacking conviction. Designed for 1h timeframe
# to capture medium-term moves while minimizing fee impact through discrete position sizing (0.20) and session filter
# (08-20 UTC) to reduce noise trades. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate previous day's Camarilla levels (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla calculations: based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use only completed previous day's data
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + range_ * 1.1 / 4
    camarilla_s3 = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # Pre-compute session hours
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for volume MA and 4h EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Check session filter
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else signals[i-1]  # Hold position outside session
            continue
            
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation, session active, and trend filter
            if curr_volume_confirm and in_session[i]:
                # Bullish breakout: price above Camarilla R3 + price above 4h EMA50
                if curr_close > camarilla_r3_aligned[i] and curr_close > curr_ema_50_4h:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price below Camarilla S3 + price below 4h EMA50
                elif curr_close < camarilla_s3_aligned[i] and curr_close < curr_ema_50_4h:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 (mean reversion) OR loses 4h EMA50 support
            if (curr_close < camarilla_s3_aligned[i] or 
                curr_close < curr_ema_50_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 (mean reversion) OR loses 4h EMA50 resistance
            if (curr_close > camarilla_r3_aligned[i] or 
                curr_close > curr_ema_50_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals