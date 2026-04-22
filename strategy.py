#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA trend filter and volume confirmation.
# Uses 1d EMA(34) for trend direction, 12h Williams %R(14) for overbought/oversold signals.
# Volume spike filter reduces false signals.
# Long in uptrend when Williams %R < -80 (oversold) + volume spike.
# Short in downtrend when Williams %R > -20 (overbought) + volume spike.
# Session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 12-37 trades/year per symbol (48-148 total) to stay within fee limits.
# Williams %R is effective in ranging markets and captures reversals in trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (price > EMA34) + Williams %R oversold + volume spike
            if (close[i] > ema_34_1d_aligned[i] and 
                williams_r[i] < -80 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price < EMA34) + Williams %R overbought + volume spike
            elif (close[i] < ema_34_1d_aligned[i] and 
                  williams_r[i] > -20 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or Williams %R mean reversion
            if position == 1:
                if (close[i] < ema_34_1d_aligned[i] or williams_r[i] > -50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema_34_1d_aligned[i] or williams_r[i] < -50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_Trend_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0