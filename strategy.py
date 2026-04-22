# 4H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp
# Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as strong support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and daily EMA34 trend filter capture
# institutional moves. Works in bull/bear because trend filter adapts direction and volume
# confirms legitimacy, reducing false breakouts. Target: 20-40 trades/year.
# Timeframe: 4h, HTF: 1d for pivots, EMA, and volume average.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's OHLC to avoid look-ahead
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_r1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    camarilla_s1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(df_daily['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above daily EMA34 (uptrend)
            if (close[i] > r1_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below daily EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_aligned[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Camarilla level
            if position == 1:
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "4h"
leverage = 1.0