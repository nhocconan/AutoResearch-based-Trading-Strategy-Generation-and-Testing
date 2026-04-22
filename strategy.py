# 4H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS
# Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as strong support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and daily EMA34 trend filter capture
# institutional moves. Works in both bull/bear markets as it follows price action with trend filter.
# Uses 4h timeframe for entries, 1d for pivots/trend/volume filter. Target: 20-50 trades/year.

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
    
    # Load daily data for Camarilla pivots, EMA34, and volume average - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_daily, 1)
    prev_low = np.roll(low_daily, 1)
    prev_close = np.roll(close_daily, 1)
    prev_high[0] = high_daily[0]  # First day uses same day's high
    prev_low[0] = low_daily[0]    # First day uses same day's low
    prev_close[0] = close_daily[0] # First day uses same day's close
    
    # Calculate Camarilla levels
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # Calculate EMA34 on daily close
    ema_34 = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period average volume on daily
    vol_avg_20 = pd.Series(df_daily['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
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
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume and trend filter
            if (close[i] > r1_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and trend filter
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

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0