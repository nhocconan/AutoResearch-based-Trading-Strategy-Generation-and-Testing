#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND close > 1w EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 level AND close < 1w EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses back inside the Camarilla H3-L3 range (mean reversion of breakout failure)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 7-25 trades/year on 1d.
# Camarilla pivot levels provide objective intraday support/resistance that work in ranging markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# 1w EMA50 filter ensures we only trade with the higher timeframe trend.
# Works in bull markets via upward breakouts from R3, works in bear via downward breakouts from S3.

name = "1d_Camarilla_R3S3_1wEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels on 1d data (using previous day's OHLC)
    # Camarilla levels: H5 = close + 1.1*(high-low)*1.1/2, H4 = close + 1.1*(high-low)*1.1/4, etc.
    # We need previous day's OHLC to calculate today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Handle first bar (no previous day)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + rang * 1.1 / 4
    camarilla_l3 = prev_close - rang * 1.1 / 4
    camarilla_h4 = prev_close + rang * 1.1 / 2
    camarilla_l4 = prev_close - rang * 1.1 / 2
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1w_aligned[i]
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        curr_close = close[i]
        prev_close_val = close[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla H3 level AND close > 1w EMA50 AND volume confirmation
            if curr_close > h3 and prev_close_val <= h3 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla L3 level AND close < 1w EMA50 AND volume confirmation
            elif curr_close < l3 and prev_close_val >= l3 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses back inside Camarilla H3-L3 range
            if curr_close < h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses back inside Camarilla H3-L3 range
            if curr_close > l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals