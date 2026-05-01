#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation
# Uses 4h EMA50 for trend direction (price > EMA50 = uptrend, price < EMA50 = downtrend)
# Entry: Long when price breaks above R3 AND volume > 1.5x 20-bar avg volume in uptrend
#        Short when price breaks below S3 AND volume > 1.5x 20-bar avg volume in downtrend
# Exit: Opposite Camarilla level break (R4/S4) or time-based exit (24 bars max hold)
# Designed for low frequency (60-150 trades over 4 years) with clear structure and volume filter

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels for 1h (based on previous bar)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Actually: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using previous bar to avoid look-ahead
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    rng = prev_high - prev_low
    r3 = prev_close + 1.1 * rng * (1.1/4)
    s3 = prev_close - 1.1 * rng * (1.1/4)
    r4 = prev_close + 1.1 * rng * (1.1/2)
    s4 = prev_close - 1.1 * rng * (1.1/2)
    
    # Volume confirmation: volume > 1.5x 20-bar average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Update bars in trade
        if position != 0:
            bars_in_trade += 1
        
        # Exit conditions
        exit_signal = False
        if position == 1:  # Long
            # Exit on R4 break (stronger resistance) or max hold time (24 bars = 1 day)
            if close[i] >= r4[i] or bars_in_trade >= 24:
                exit_signal = True
        elif position == -1:  # Short
            # Exit on S4 break (stronger support) or max hold time
            if close[i] <= s4[i] or bars_in_trade >= 24:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            bars_in_trade = 0
            continue
        
        # Entry logic (only when flat)
        if position == 0 and in_session[i]:
            # Uptrend: price > 4h EMA50
            if close[i] > ema50_4h_aligned[i]:
                # Long: break above R3 with volume confirmation
                if close[i] > r3[i] and vol_confirm[i]:
                    signals[i] = 0.20
                    position = 1
                    bars_in_trade = 1
            # Downtrend: price < 4h EMA50
            elif close[i] < ema50_4h_aligned[i]:
                # Short: break below S3 with volume confirmation
                if close[i] < s3[i] and vol_confirm[i]:
                    signals[i] = -0.20
                    position = -1
                    bars_in_trade = 1
        
        # Hold position
        if position == 1 and not exit_signal:
            signals[i] = 0.20
        elif position == -1 and not exit_signal:
            signals[i] = -0.20
    
    return signals