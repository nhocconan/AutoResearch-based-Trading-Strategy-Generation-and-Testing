#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide high-probability intraday support/resistance derived from prior day's range.
# R3 (resistance 3) and S3 (support 3) are strong levels where price often reverses or accelerates.
# Breakout above R3 with volume spike and 1d EMA34 uptrend = long signal.
# Breakout below S3 with volume spike and 1d EMA34 downtrend = short signal.
# Uses 12h timeframe to target 12-37 trades/year (50-150 total over 4 years) minimizing fee drag.
# Volume confirmation ensures institutional participation. Works in bull (breakouts with volume) and bear
# (mean reversion after volatility expansion via opposite breakouts).

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    # Camarilla: H1 = prior day high, L1 = prior day low, C1 = prior day close
    # R3 = C1 + (H1 - L1) * 1.1/2, S3 = C1 - (H1 - L1) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    camarilla_R3 = close_1d_vals + (high_1d - low_1d) * 1.1 / 2
    camarilla_S3 = close_1d_vals - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation: current volume > 2.0 * 24-period average volume (12h * 2 = 1d)
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 34  # Need sufficient history for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_close > camarilla_R3_aligned[i]  # Break above R3
        breakout_down = curr_close < camarilla_S3_aligned[i]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Camarilla R3, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Camarilla S3, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below Camarilla S3 or trend reversal
            if curr_close < camarilla_S3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above Camarilla R3 or trend reversal
            if curr_close > camarilla_R3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals