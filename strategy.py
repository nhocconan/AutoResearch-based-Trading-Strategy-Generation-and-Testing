#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND close > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 level AND close < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses back inside the Camarilla H3/L3 levels (mean reversion of breakout failure)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# Camarilla pivot levels provide objective intraday support/resistance that work in ranging and trending markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# 1d EMA34 filter ensures we only trade with the higher timeframe trend.
# Works in bull markets via upward breakouts, works in bear via downward breakouts with volume spikes.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels on 1d data (using previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # We'll use daily OHLC to calculate these levels
    ohlc_1d = df_1d[['open', 'high', 'low', 'close']].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.zeros(len(ohlc_1d))
    camarilla_s3 = np.zeros(len(ohlc_1d))
    camarilla_h3 = np.zeros(len(ohlc_1d))
    camarilla_l3 = np.zeros(len(ohlc_1d))
    
    for i in range(len(ohlc_1d)):
        o, h, l, c = ohlc_1d[i]
        rng = h - l
        camarilla_r3[i] = c + (rng * 1.1 / 2)
        camarilla_s3[i] = c - (rng * 1.1 / 2)
        camarilla_h3[i] = c + (rng * 1.1 / 4)
        camarilla_l3[i] = c - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_34_1d_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        curr_close = close[i]
        prev_close = close[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume confirmation
            if curr_close > r3 and prev_close <= r3 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume confirmation
            elif curr_close < s3 and prev_close >= s3 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses back inside Camarilla H3/L3 levels
            if curr_close < h3 and curr_close > l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses back inside Camarilla H3/L3 levels
            if curr_close < h3 and curr_close > l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals