#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide precise support/resistance from prior day's range
# Entry: Long when price breaks above R3 with bullish 1d trend and volume confirmation
# Entry: Short when price breaks below S3 with bearish 1d trend and volume confirmation
# Exit: Opposite Camarilla level (S3 for long, R3 for short) or trend reversal
# Uses discrete sizing (0.25) to control fee drag. Target: 75-200 total trades over 4 years.
# Camarilla breakouts work in both bull/bear markets by capturing institutional levels.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate 1d Camarilla levels (based on prior 1d candle's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1day OHLC for Camarilla calculation
    prior_high = df_1d['high'].shift(1).values  # Shift to use completed 1d bar
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    prior_range = prior_high - prior_low
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prior_close + prior_range * 1.1 / 4
    camarilla_s3 = prior_close - prior_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 35  # Need enough data for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 with bullish trend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with bearish trend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 OR trend turns bearish
            if (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 OR trend turns bullish
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals