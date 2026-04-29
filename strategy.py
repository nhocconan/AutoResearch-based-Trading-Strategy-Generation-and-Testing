#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2x 24-bar average
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2x 24-bar average
# Exit when price retouches the 1d EMA34 (mean reversion to trend) or opposite Camarilla level
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.
# Camarilla levels provide intraday support/resistance based on prior day's range.
# 1d EMA34 filters counter-trend moves, volume confirmation ensures institutional participation.
# Works in bull markets (buying R3 breakouts) and bear markets (selling S3 breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for Camarilla calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    # H1 = prior day high, L1 = prior day low, C1 = prior day close
    H1 = df_1d['high'].shift(1).values  # Prior day high
    L1 = df_1d['low'].shift(1).values   # Prior day low
    C1 = df_1d['close'].shift(1).values # Prior day close
    
    # Camarilla R3 = C1 + (H1 - L1) * 1.1/4
    # Camarilla S3 = C1 - (H1 - L1) * 1.1/4
    R3 = C1 + (H1 - L1) * 1.1 / 4
    S3 = C1 - (H1 - L1) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for prior day close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: >2x 24-bar average volume (24*12h = 12 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # EMA34 warmup and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_34 = ema_34_1d_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retouches 1d EMA34 (mean reversion) or breaks below S3
            if close[i] <= ema_34 or close[i] < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retouches 1d EMA34 (mean reversion) or breaks above R3
            if close[i] >= ema_34 or close[i] > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 1d EMA34 AND volume confirmation
            if close[i] > r3 and close[i] > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND close < 1d EMA34 AND volume confirmation
            elif close[i] < s3 and close[i] < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals