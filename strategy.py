#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance, EMA50 on 4h ensures alignment with intermediate trend
# Volume confirmation filters false breakouts. Designed for 1h timeframe targeting 15-37 trades/year (60-150 total over 4 years)
# Uses discrete position sizing (0.20) to balance return and drawdown control
# Works in bull markets (breakout above R1 + 4h EMA50 up) and bear markets (breakout below S1 + 4h EMA50 down)
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_Volume_Session"
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
    
    # Session filter: 08-20 UTC (avoid low-liquidity Asian session)
    hours = open_time.dt.hour.values
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter (EMA50) and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 4h EMA50 calculation
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots from previous 4h bar (OHLC)
    # Camarilla levels: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1 for each 4h bar
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Camarilla R1 with volume confirmation and uptrend
            if high[i] > camarilla_r1_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: Breakout below Camarilla S1 with volume confirmation and downtrend
            elif low[i] < camarilla_s1_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S1 (reversal) OR trend changes
            if low[i] < camarilla_s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R1 (reversal) OR trend changes
            if high[i] > camarilla_r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals