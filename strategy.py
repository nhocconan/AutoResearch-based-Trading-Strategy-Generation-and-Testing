#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot R3/S3 breakout with 1w EMA34 trend filter and volume spike (>1.5x 20-bar MA)
# Camarilla pivots identify key intraday support/resistance levels. R3/S3 are strong breakout levels.
# Long when price breaks above R3 with volume spike and price above 1w EMA34 (uptrend)
# Short when price breaks below S3 with volume spike and price below 1w EMA34 (downtrend)
# Uses 1w EMA34 for higher-timeframe trend alignment to reduce whipsaws in ranging markets.
# Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) on 1w close
    ema_1w_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Based on previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current close as previous
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + (range_val * 1.1 / 4)
    camarilla_s3 = prev_close - (range_val * 1.1 / 4)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3, volume spike, price above 1w EMA34 (uptrend)
            if curr_high > camarilla_r3[i] and vol_spike and curr_close > ema_1w_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, volume spike, price below 1w EMA34 (downtrend)
            elif curr_low < camarilla_s3[i] and vol_spike and curr_close < ema_1w_34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below S3 (reversal) or below 1w EMA34 (trend change)
            if curr_low < camarilla_s3[i] or curr_close < ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above R3 (reversal) or above 1w EMA34 (trend change)
            if curr_high > camarilla_r3[i] or curr_close > ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals