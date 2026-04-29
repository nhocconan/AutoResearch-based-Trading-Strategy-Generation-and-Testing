#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price retouches Camarilla pivot point (PP) or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year on 12h.
# Camarilla pivot levels provide intraday support/resistance with proven edge in ranging markets.
# 1d EMA34 filter ensures we only trade with the long-term trend, improving win rate in both bull/bear.
# Volume confirmation ensures breakouts have conviction, reducing false signals.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 12h bar (H1, L1, C1)
    # Using prior bar to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    H1 = high_series.shift(1)  # prior 12h high
    L1 = low_series.shift(1)   # prior 12h low
    C1 = close_series.shift(1) # prior 12h close
    
    PP = (H1 + L1 + C1) / 3.0
    R3 = PP + (H1 - L1) * 1.1 / 2.0
    S3 = PP - (H1 - L1) * 1.1 / 2.0
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # volume MA needs 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(PP[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        pp = PP[i]
        r3 = R3[i]
        s3 = S3[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume confirmation
            if curr_high > r3 and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume confirmation
            elif curr_low < s3 and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retouches PP or breaks below S3
            if curr_close <= pp or curr_low < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retouches PP or breaks above R3
            if curr_close >= pp or curr_high > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals