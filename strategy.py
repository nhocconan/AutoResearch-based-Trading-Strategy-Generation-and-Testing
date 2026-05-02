#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal/breakout points
# Breakout occurs when price closes outside R3/S3 with volume > 1.5x average
# Trend filter: 4h EMA50 - only trade in direction of higher timeframe trend
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Target: 80-120 total trades over 4 years (20-30/year) with discrete sizing 0.20
# Works in bull markets by capturing breakouts, in bear markets by fading false breaks

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate prior period's high, low, close for Camarilla pivots
    # Using 1h chart, so prior period is previous 1h bar
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance levels
    R3 = pivot + (range_hl * 1.1 / 4.0)
    R4 = pivot + (range_hl * 1.1 / 2.0)
    
    # Support levels
    S3 = pivot - (range_hl * 1.1 / 4.0)
    S4 = pivot - (range_hl * 1.1 / 2.0)
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close above R3 with bullish trend and volume spike
            if (close[i] > R3[i] and 
                close[i-1] <= R3[i-1] and  # Just broke above
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: Close below S3 with bearish trend and volume spike
            elif (close[i] < S3[i] and 
                  close[i-1] >= S3[i-1] and  # Just broke below
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below R3 (mean reversion) OR trend turns bearish
            if close[i] < R3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price closes above S3 (mean reversion) OR trend turns bullish
            if close[i] > S3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals