#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level with 12h EMA34 uptrend and volume > 1.8x 24-bar average
# Short when price breaks below Camarilla S3 level with 12h EMA34 downtrend and volume > 1.8x 24-bar average
# Exit via close-based reversal: long exit when price closes below Camarilla pivot point (PP)
#                      short exit when price closes above Camarilla pivot point (PP)
# Uses Camarilla pivot levels from 6h timeframe for structure, 12h EMA34 for trend filter, volume for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 75-150 total trades over 4 years = 19-37/year.

name = "6h_Camarilla_R3S3_12hEMA34_Volume_CloseExit_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla pivot levels (using prior 6h bar's OHLC)
    ph = high.shift(1).values  # prior 6h high
    pl = low.shift(1).values   # prior 6h low
    pc = close.shift(1).values # prior 6h close
    
    # Camarilla levels: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_pp = (ph + pl + pc) / 3
    camarilla_r3 = pc + (ph - pl) * 1.1 / 4
    camarilla_s3 = pc - (ph - pl) * 1.1 / 4
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation (1.8x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA34 and volume calculations)
    start_idx = 34  # EMA34 needs 34 bars
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pp[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with 12h EMA34 uptrend and volume spike
            if close[i] > camarilla_r3[i] and ema_34_aligned[i] > ema_34_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 with 12h EMA34 downtrend and volume spike
            elif close[i] < camarilla_s3[i] and ema_34_aligned[i] < ema_34_aligned[i-1] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price closes below Camarilla pivot point (PP)
            if close[i] < camarilla_pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above Camarilla pivot point (PP)
            if close[i] > camarilla_pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals