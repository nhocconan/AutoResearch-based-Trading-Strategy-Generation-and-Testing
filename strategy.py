#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels from prior 12h session to identify key support/resistance.
# Breakouts above R3 or below S3 with volume spike and alignment with 12h EMA34 trend.
# Designed for 12-30 trades/year on 6h to minimize fee drag while capturing institutional moves.
# Works in bull markets via R3 breakouts in uptrend and bear markets via S3 breakdowns in downtrend.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Camarilla pivots and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from prior 12h bar (HLC of completed 12h bar)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    for i in range(1, len(df_12h)):
        # Prior completed 12h bar
        phigh = df_12h['high'].iloc[i-1]
        plow = df_12h['low'].iloc[i-1]
        pclose = df_12h['close'].iloc[i-1]
        range_ = phigh - plow
        if range_ <= 0:
            camarilla_r3[i] = camarilla_s3[i] = pclose
        else:
            camarilla_r3[i] = pclose + range_ * 1.1 / 4
            camarilla_s3[i] = pclose - range_ * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (wait for 12h bar close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after sufficient warmup for EMA34
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: bullish breakout above R3 in 12h uptrend with volume spike
            if breakout_up and ema_34_12h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown below S3 in 12h downtrend with volume spike
            elif breakout_down and ema_34_12h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of R3-S3 or loses 12h uptrend
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] < midpoint or ema_34_12h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of R3-S3 or loses 12h downtrend
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] > midpoint or ema_34_12h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals