#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide precise support/resistance. R3/S3 breakouts with volume spike
# indicate institutional interest. 1w EMA34 ensures we trade with the weekly trend.
# Works in bull (breakouts with volume) and bear (trend continuation after pullbacks to EMA).
# Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA34 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) calculation
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1w HTF data for Camarilla pivot calculation (using prior week's OHLC)
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla formulas: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_high = close_1w + 1.1 * (high_1w - low_1w)  # R3
    camarilla_low = close_1w - 1.1 * (high_1w - low_1w)   # S3
    
    # Align Camarilla levels to 1d timeframe (using prior week's levels)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1w, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1w, camarilla_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 34  # Need 34 for EMA + 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Camarilla breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > camarilla_high_aligned[i-1]  # Break above R3
        breakout_down = curr_close < camarilla_low_aligned[i-1]  # Break below S3
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike[i]
        # Trend filter: price above/below 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakdown or trend reversal
            if curr_close < camarilla_low_aligned[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout or trend reversal
            if curr_close > camarilla_high_aligned[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals