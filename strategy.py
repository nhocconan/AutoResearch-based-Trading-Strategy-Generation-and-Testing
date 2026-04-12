# 12h_1d_4h_Camarilla_Pivot_Breakout_V1
# Hypothesis: On 12h timeframe, trade breakouts from Camarilla pivot levels calculated from 1d candles.
# Long when price breaks above H3 with volume confirmation and 4h uptrend (price > 4h EMA20).
# Short when price breaks below L3 with volume confirmation and 4h downtrend (price < 4h EMA20).
# Uses volume > 1.5x 20-period average for confirmation.
# Designed for low trade frequency (15-35 trades/year) with strict entry conditions.
# Works in bull markets via long breakouts and in bear markets via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_4h_Camarilla_Pivot_Breakout_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d candle
    # H4 = C + 1.5*(H-L), H3 = C + 1.25*(H-L), H2 = C + 1.166*(H-L), H1 = C + 1.083*(H-L)
    # L1 = C - 1.083*(H-L), L2 = C - 1.166*(H-L), L3 = C - 1.25*(H-L), L4 = C - 1.5*(H-L)
    # where C = (H+L+CLOSE)/3 (typical price)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot levels for previous day
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    h3 = pivot + 1.25 * range_hl
    l3 = pivot - 1.25 * range_hl
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Load 4h data ONCE for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_long = close[i] > h3_aligned[i]
        breakout_short = close[i] < l3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter from 4h EMA20
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Entry conditions
        long_entry = breakout_long and volume_confirm and uptrend
        short_entry = breakout_short and volume_confirm and downtrend
        
        # Exit conditions: price returns to pivot level or opposite breakout
        pivot_level = (h3_aligned[i] + l3_aligned[i]) / 2
        long_exit = close[i] < pivot_level
        short_exit = close[i] > pivot_level
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals