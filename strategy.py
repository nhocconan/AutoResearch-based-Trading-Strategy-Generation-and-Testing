#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 Breakout + 1w EMA34 Trend + Volume Spike
# Camarilla pivot levels from prior day provide strong support/resistance.
# Breakout above R3 (resistance 3) with 1w EMA34 uptrend and volume spike = long.
# Breakdown below S3 (support 3) with 1w EMA34 downtrend and volume spike = short.
# Exit on retracement to pivot point (PP) or opposite Camarilla level (S3/R3).
# Uses 1w timeframe for trend filter to capture major market direction.
# Volume confirmation filters weak breakouts. Discrete position sizing (0.25) limits drawdown.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter and Camarilla calculation (requires weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from prior 1w bar (which represents prior week for 1d chart)
    # Camarilla uses prior period's OHLC: PP = (H+L+C)/3
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We shift by 1 to use prior completed 1w bar's OHLC
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    prior_close = df_1w['close'].shift(1).values
    
    # Calculate Camarilla levels
    pp = (prior_high + prior_low + prior_close) / 3.0
    r3 = prior_close + (prior_high - prior_low) * 1.1 / 4.0
    s3 = prior_close - (prior_high - prior_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 1d (they change only when 1w bar closes)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA trend filter
        ema_trend_up = close[i] > ema_34_1w_aligned[i]
        ema_trend_down = close[i] < ema_34_1w_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > R3, 1w EMA34 uptrend, volume confirm
            if price > r3_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < S3, 1w EMA34 downtrend, volume confirm
            elif price < s3_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to PP or below S3
            if price < pp_aligned[i] or price < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to PP or above R3
            if price > pp_aligned[i] or price > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals