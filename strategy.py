#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (EMA50) and volume confirmation
# Camarilla pivot levels provide high-probability breakout zones from prior 1h bar
# 4h EMA50 trend filter ensures we trade in direction of higher timeframe momentum
# Volume spike confirms institutional participation behind the breakout
# Designed for low frequency (60-150 trades over 4 years) to minimize fee drag on 1h timeframe
# Uses 4h for signal direction, 1h only for entry timing precision

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 calculation (trend filter)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate Camarilla levels from prior 1h bar (using prior bar's HLC)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    prior_high = np.concatenate([[np.nan], high[:-1]])  # prior bar's high
    prior_low = np.concatenate([[np.nan], low[:-1]])    # prior bar's low
    prior_close = np.concatenate([[np.nan], close[:-1]]) # prior bar's close
    
    hl_range = prior_high - prior_low
    camarilla_r3 = prior_close + hl_range * 1.1 / 4
    camarilla_s3 = prior_close - hl_range * 1.1 / 4
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need 4h EMA50 and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or 
            np.isnan(prior_close[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3[i]  # Price breaks above R3
        breakout_short = close[i] < camarilla_s3[i]  # Price breaks below S3
        
        # Trend filter: price above/below 4h EMA50 indicates trend direction
        # Long bias when price > EMA, short bias when price < EMA
        trend_long = close[i] > ema_4h_aligned[i]
        trend_short = close[i] < ema_4h_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 with volume spike and 4h uptrend
            if breakout_long and vol_spike and trend_long:
                signals[i] = 0.20
                position = 1
            # Short: Breakout below S3 with volume spike and 4h downtrend
            elif breakout_short and vol_spike and trend_short:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below prior bar's low or trend reversal
            if close[i] < prior_low[i] or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on close above prior bar's high or trend reversal
            if close[i] > prior_high[i] or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals