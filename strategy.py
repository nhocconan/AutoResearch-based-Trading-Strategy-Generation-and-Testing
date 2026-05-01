#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with volume confirmation (>1.8x 20-bar volume MA) and 4h EMA50 trend filter
# Camarilla levels provide precise intraday support/resistance; volume confirms breakout strength.
# 4h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise during low-activity periods.
# Discrete sizing (0.20) minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Works in bull (breakouts with volume) and bear (failed reversals at strong levels).

name = "1h_Camarilla_R3S3_Breakout_VolumeSpike_4hEMA50_Trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) on 4h close
    ema_4h_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Camarilla levels calculation (based on prior day's range)
    # Need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_day_close = df_1d['close'].shift(1).values  # Shift to avoid look-ahead
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3, R4, S3, S4 levels
    # R4 = Close + 1.1*(High-Low)*1.1/2
    # R3 = Close + 1.1*(High-Low)*1.1/4
    # S3 = Close - 1.1*(High-Low)*1.1/4
    # S4 = Close - 1.1*(High-Low)*1.1/2
    camarilla_r3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) * 1.1 / 4
    camarilla_r4 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) * 1.1 / 2
    camarilla_s3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) * 1.1 / 4
    camarilla_s4 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (with 1-day delay for completed bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3, additional_delay_bars=1)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3, additional_delay_bars=1)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4, additional_delay_bars=1)
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for 4h EMA and daily data propagation
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Camarilla breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > camarilla_r3_aligned[i-1]  # Break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i-1]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up, price above 4h EMA50, volume spike
            if breakout_up and curr_close > ema_4h_50_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla breakout down, price below 4h EMA50, volume spike
            elif breakout_down and curr_close < ema_4h_50_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakdown below S3 or price below 4h EMA50
            if curr_close < camarilla_s3_aligned[i] or curr_close < ema_4h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout above R3 or price above 4h EMA50
            if curr_close > camarilla_r3_aligned[i] or curr_close > ema_4h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals