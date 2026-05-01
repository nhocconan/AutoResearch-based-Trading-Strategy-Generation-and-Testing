#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide high-probability intraday support/resistance.
# Breakout of R3/S3 with 1d EMA trend alignment captures strong momentum moves.
# Volume confirmation ensures institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) calculation for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use R3 and S3 levels (R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4))
    camarilla_R3 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 4
    camarilla_S3 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3.values)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for EMA(34) + 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = curr_close > ema_34_1d_aligned[i]
        trend_down = curr_close < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > camarilla_R3_aligned[i-1]  # Break above R3
        breakout_down = curr_close < camarilla_S3_aligned[i-1]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up (above R3), volume spike, uptrend
            if breakout_up and vol_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down (below S3), volume spike, downtrend
            elif breakout_down and vol_spike and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla reversion (below R3) or trend reversal
            if curr_close < camarilla_R3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Camarilla reversion (above S3) or trend reversal
            if curr_close > camarilla_S3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals