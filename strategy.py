#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND close > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 AND close < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses back inside Camarilla H3/L3 levels (mean reversion)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-50 trades/year on 4h.
# Camarilla levels from daily timeframe provide high-probability intraday reversal/breakout zones.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# 1d EMA50 filter ensures we only trade with the higher timeframe trend.
# Works in bull markets via upward breakouts at resistance, works in bear via downward breakouts at support.

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla equations:
    # H4 = close + 1.1*(high - low)/2
    # H3 = close + 1.1*(high - low)/4
    # H2 = close + 1.1*(high - low)/6
    # H1 = close + 1.1*(high - low)/12
    # L1 = close - 1.1*(high - low)/12
    # L2 = close - 1.1*(high - low)/6
    # L3 = close - 1.1*(high - low)/4
    # L4 = close - 1.1*(high - low)/2
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_h2 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l2 = close_1d - 1.1 * (high_1d - low_1d) / 6
    camarilla_h1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_l1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        h2 = camarilla_h2_aligned[i]
        l2 = camarilla_l2_aligned[i]
        h1 = camarilla_h1_aligned[i]
        l1 = camarilla_l1_aligned[i]
        curr_close = close[i]
        prev_close = close[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla H3 AND close > 1d EMA50 AND volume confirmation
            if curr_close > h3 and prev_close <= h3 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla L3 AND close < 1d EMA50 AND volume confirmation
            elif curr_close < l3 and prev_close >= l3 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses back inside H2/L2 (mean reversion)
            if curr_close < h2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses back inside L2/H2 (mean reversion)
            if curr_close > l2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals