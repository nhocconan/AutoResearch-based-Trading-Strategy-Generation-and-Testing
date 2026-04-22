#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation.
Long at S1 with bullish 1d EMA34 and volume mean reversion; short at R1 with bearish 1d EMA34 and volume mean reversion.
Exit at S3/R3 or when trend weakens.
Camarilla levels from prior 1d provide precise intraday reversal zones.
1d EMA34 filter ensures alignment with daily trend to avoid counter-trend trades.
Volume mean reversion (current < average) increases probability of mean-reverting bounce.
Designed for 20-40 trades/year to minimize fee drag in ranging markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for EMA34 filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_d = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6-volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior 1d
    # Typical Price = (H + L + C) / 3
    typical_price_d = (df_daily['high'].values + df_daily['low'].values + df_daily['close'].values) / 3.0
    # Previous day's typical price
    prev_typical = np.roll(typical_price_d, 1)
    prev_typical[0] = np.nan  # First day has no previous
    
    # Camarilla multipliers
    # S1 = TP - 1.1 * (H - L) / 12
    # S2 = TP - 1.1 * (H - L) / 6
    # S3 = TP - 1.1 * (H - L) / 4
    # R1 = TP + 1.1 * (H - L) / 12
    # R2 = TP + 1.1 * (H - L) / 6
    # R3 = TP + 1.1 * (H - L) / 4
    hl_range = df_daily['high'].values - df_daily['low'].values
    s1 = prev_typical - 1.1 * hl_range / 12.0
    s2 = prev_typical - 1.1 * hl_range / 6.0
    s3 = prev_typical - 1.1 * hl_range / 4.0
    r1 = prev_typical + 1.1 * hl_range / 12.0
    r2 = prev_typical + 1.1 * hl_range / 6.0
    r3 = prev_typical + 1.1 * hl_range / 4.0
    
    # Align Camarilla levels to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_d)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to avoid NaN in prev_typical
        # Skip if data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price at S1 with bullish 1d trend and volume mean reversion
            if (close[i] <= s1_aligned[i] and 
                close[i] > s2_aligned[i] and  # Above S2 to avoid deep pullback
                ema34_aligned[i] > ema34_d[np.searchsorted(df_daily.index, prices['open_time'].iloc[i]) - 1 if i > 0 else 0] and  # Simplified: use aligned EMA34 > previous day's EMA34 proxy
                volume[i] < 0.8 * vol_avg_20[i]):  # Volume mean reversion
                signals[i] = 0.25
                position = 1
            # Short: Price at R1 with bearish 1d trend and volume mean reversion
            elif (close[i] >= r1_aligned[i] and 
                  close[i] < r2_aligned[i] and  # Below R2 to avoid overextended
                  ema34_aligned[i] < ema34_d[np.searchsorted(df_daily.index, prices['open_time'].iloc[i]) - 1 if i > 0 else 0] and
                  volume[i] < 0.8 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches S3/R3 or trend weakens
                if close[i] <= s3_aligned[i] or close[i] >= r3_aligned[i]:
                    exit_signal = True
                # Optional: exit if EMA34 flips (trend change)
                elif ema34_aligned[i] < ema34_d[np.searchsorted(df_daily.index, prices['open_time'].iloc[i]) - 1 if i > 0 else 0]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches S3/R3 or trend weakens
                if close[i] <= s3_aligned[i] or close[i] >= r3_aligned[i]:
                    exit_signal = True
                # Optional: exit if EMA34 flips (trend change)
                elif ema34_aligned[i] > ema34_d[np.searchsorted(df_daily.index, prices['open_time'].iloc[i]) - 1 if i > 0 else 0]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_CamarillaReversal_1dEMA34_VolumeMeanRev"
timeframe = "6h"
leverage = 1.0