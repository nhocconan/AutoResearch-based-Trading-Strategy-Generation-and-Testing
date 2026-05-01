#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout + 1w EMA34 Trend + Volume Spike (>1.8x 20-bar volume MA)
# Camarilla pivot levels provide high-probability intraday reversal/breakout points.
# Breakout above R3 or below S3 with volume confirmation and aligned weekly trend captures strong moves.
# Weekly EMA34 filter ensures we trade with the higher-timeframe trend, reducing false breakouts.
# Volume spike (>1.8x 20-bar MA) confirms institutional participation.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(34) on 1w close
    ema_1w_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # 1d data for Camarilla pivot calculation (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use the prior day's OHLC to avoid look-ahead
    prior_close = df_1d['close'].shift(1).values  # prior day close
    prior_high = df_1d['high'].shift(1).values    # prior day high
    prior_low = df_1d['low'].shift(1).values      # prior day low
    
    # Calculate Camarilla R3 and S3 levels
    rang = prior_high - prior_low
    camarilla_r3 = prior_close + 1.1 * rang
    camarilla_s3 = prior_close - 1.1 * rang
    
    # Align Camarilla levels to 12h timeframe (1d -> 12h: 2 bars per day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 34  # Need 34 for 1w EMA and 1 for prior day data (34 > 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Breakout conditions
        breakout_long = curr_close > camarilla_r3_aligned[i]
        breakout_short = curr_close < camarilla_s3_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Weekly trend filter
        weekly_uptrend = ema_1w_34_aligned[i] < close[i]  # price above weekly EMA = uptrend
        weekly_downtrend = ema_1w_34_aligned[i] > close[i]  # price below weekly EMA = downtrend
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3, volume spike, weekly uptrend
            if breakout_long and vol_spike and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3, volume spike, weekly downtrend
            elif breakout_short and vol_spike and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakout below S3 or weekly trend turns down
            if curr_close < camarilla_s3_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R3 or weekly trend turns up
            if curr_close > camarilla_r3_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals