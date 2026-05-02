#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance, 4h EMA34 ensures alignment with intermediate trend
# Volume confirmation filters false breakouts. Designed for 1h timeframe targeting 15-37 trades/year (60-150 total over 4 years)
# Uses discrete position sizing (0.20) to balance return and drawdown control
# Works in bull markets (breakout above R3 + 4h EMA34 up) and bear markets (breakout below S3 + 4h EMA34 down)
# Session filter (08-20 UTC) reduces noise and avoids Asian session chop

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_Volume"
timeframe = "1h"
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
    
    # Precompute session hours for 08-20 UTC filter
    hours = open_time.dt.hour.values
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter (EMA34) and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 4h EMA34 calculation
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels from previous 4h bar (OHLC)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    camarilla_upper = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_lower = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_4h, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_4h, camarilla_lower)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_upper_aligned[i]) or 
            np.isnan(camarilla_lower_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA34
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Camarilla R3 with volume confirmation and uptrend
            if high[i] > camarilla_upper_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: Breakout below Camarilla S3 with volume confirmation and downtrend
            elif low[i] < camarilla_lower_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 (reversal) OR trend changes
            if low[i] < camarilla_lower_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 (reversal) OR trend changes
            if high[i] > camarilla_upper_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals