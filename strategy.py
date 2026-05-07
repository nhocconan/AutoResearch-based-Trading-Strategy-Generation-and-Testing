#1.0.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal with 1-day trend filter (EMA34) and volume spike confirmation.
# Long when: Price touches Camarilla S1 (support) AND EMA34(1d) rising AND volume > 2.0 * EMA20(volume).
# Short when: Price touches Camarilla R1 (resistance) AND EMA34(1d) falling AND volume > 2.0 * EMA20(volume).
# Exit when price reverses to touch Camarilla H4/L4 levels.
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag and improve generalization.
# Works in bull markets via S1 bounces and in bear markets via R1 rejections.
name = "12h_Camarilla_S1R1_1dEMA34_Volume"
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
    
    # Calculate Camarilla levels from previous 12h bar
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # R1 = close + 1.1 * (high - low)
    # S1 = close - 1.1 * (high - low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_H4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_L4 = prev_close - 1.5 * (prev_high - prev_low)
    camarilla_R1 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_S1 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Rising if current > previous, falling if current < previous
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i]) or np.isnan(camarilla_R1[i]) or 
            np.isnan(camarilla_S1[i]) or np.isnan(ema_34_rising_aligned[i]) or 
            np.isnan(ema_34_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches S1 AND EMA34(1d) rising AND volume spike
            long_condition = (low[i] <= camarilla_S1[i]) and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Price touches R1 AND EMA34(1d) falling AND volume spike
            short_condition = (high[i] >= camarilla_R1[i]) and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price touches H4 (resistance level)
            if high[i] >= camarilla_H4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price touches L4 (support level)
            if low[i] <= camarilla_L4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals