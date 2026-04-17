# 12h_Camarilla_Pivot_Breakout_1dEMA200_Volume
# Hypothesis: 12h strategy using Camarilla pivot levels (R1, S1) from 1-day data combined with
# 1-day EMA200 trend filter and volume confirmation. Enter long when price breaks above R1
# with volume > 1.5x 20-period volume MA and price above 1-day EMA200. Enter short when
# price breaks below S1 with volume > 1.5x 20-period volume MA and price below 1-day EMA200.
# Exit when price returns to the Camarilla pivot point (PP). Designed for 12h timeframe
# with strict entry conditions to limit trades to 50-150 total over 4 years.
# Works in both bull and bear markets by following the 1-day EMA200 trend and using
# volatility-based Camarilla levels that adapt to recent price action.

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
    
    # Get 1-day data for Camarilla pivot calculation and EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Camarilla pivot levels from previous 1-day data
    # Typical Camarilla formula using previous day's OHLC
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    # We need previous day's OHLC, so we shift by 1 to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each 1-day bar
    # Avoid division by zero or invalid calculations
    hl_range = prev_high - prev_low
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = prev_close + (hl_range * 1.0833)
    s1 = prev_close - (hl_range * 1.0833)
    
    # Align Camarilla levels to 12h timeframe (wait for 1-day bar to complete)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: 20-period volume MA on 12h data
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        ema_val = ema_200_aligned[i]
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend filter
            # Long: price breaks above R1, volume spike, price above EMA200
            if price > r1_val and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, volume spike, price below EMA200
            elif price < s1_val and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to pivot point (mean reversion)
            if price <= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to pivot point (mean reversion)
            if price >= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0