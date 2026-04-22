#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and 1d volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaws: 13-period SMMA, Teeth: 8-period, Lips: 5-period)
# When the three lines are intertwined (alligator sleeping), market is ranging.
# When they diverge (alligator waking up), a trend is forming.
# Direction determined by the order of the lines: if Lips > Teeth > Jaws = uptrend, reverse for downtrend.
# Entry confirmed by 1d volume spike (> 1.5x 20-day average) to avoid false signals.
# Works in bull markets by capturing uptrends and in bear markets by capturing downtrends.
# Designed for 12h timeframe targeting 12-37 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator components (using 12h data)
    # Jaws: 13-period SMMA shifted 8 bars
    jaws = pd.Series(close).rolling(window=13, center=False).mean().shift(8).values
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, center=False).mean().shift(5).values
    # Lips: 5-period SMMA shifted 3 bars
    lips = pd.Series(close).rolling(window=5, center=False).mean().shift(3).values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaws (alligator awake, uptrend) + 1d uptrend + 1d volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaws[i] and
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (alligator awake, downtrend) + 1d downtrend + 1d volume spike
            elif (jaws[i] > teeth[i] and teeth[i] > lips[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: alligator sleeping (lines intertwined) or trend reversal
            if position == 1:
                # Exit on alligator sleeping or trend reversal
                if (lips[i] <= teeth[i] or teeth[i] <= jaws[i] or
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on alligator sleeping or trend reversal
                if (teeth[i] <= lips[i] or jaws[i] <= teeth[i] or
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_1dVolSpike"
timeframe = "12h"
leverage = 1.0