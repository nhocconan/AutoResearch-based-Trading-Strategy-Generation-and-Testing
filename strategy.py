#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Long when price breaks above 1d Camarilla R3 AND close > 1w EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below 1d Camarilla S3 AND close < 1w EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses back inside the Camarilla H3/L3 levels (mean reversion of breakout failure)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 7-25 trades/year on 1d.
# Camarilla levels provide objective breakout levels that work in both trending and ranging markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# 1w EMA34 filter ensures we only trade with the higher timeframe trend.
# Works in bull markets via upward breakouts, works in bear via downward breakouts with volume spikes.

name = "1d_Camarilla_R3S3_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: H4 = C + 1.1*(H-L)/2, H3 = C + 1.1*(H-L)/4, etc.
    # We need previous day's data, so we shift by 1
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Calculate Camarilla levels
    camarilla_range = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * camarilla_range / 4  # R3
    camarilla_l3 = prev_close - 1.1 * camarilla_range / 4  # S3
    camarilla_h4 = prev_close + 1.1 * camarilla_range / 2  # Used for exit
    camarilla_l4 = prev_close - 1.1 * camarilla_range / 2  # Used for exit
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need sufficient history for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_l4[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_34_1w_aligned[i]
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h4 = camarilla_h4[i]
        l4 = camarilla_l4[i]
        curr_close = close[i]
        prev_close_val = close[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla H3 (R3) AND close > 1w EMA34 AND volume confirmation
            if curr_close > h3 and prev_close_val <= h3 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla L3 (S3) AND close < 1w EMA34 AND volume confirmation
            elif curr_close < l3 and prev_close_val >= l3 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses back inside Camarilla H4/L4
            if curr_close < h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses back inside Camarilla H4/L4
            if curr_close > l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals