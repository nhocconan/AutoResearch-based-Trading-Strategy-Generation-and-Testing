#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(34) trend filter and volume confirmation (>2.0x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 1h bar for structure (breakout = new R3/S3 level)
# 4h EMA(34) filter ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation ensures breakout has sufficient participation
# Discrete sizing 0.20 balances risk and return while minimizing fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# BTC/ETH focus: avoids SOL-only bias by requiring HTF trend alignment

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # Calculate 4h EMA(34) trend filter from prior completed 4h bar
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_shifted = np.roll(ema_34_4h, 1)
    ema_34_4h_shifted[0] = np.nan
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h_shifted)
    
    # Calculate Camarilla pivot levels for 1h timeframe (using prior completed 1h bar)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6)
    #          S4 = C - ((H-L)*1.1/2), S3 = C - ((H-L)*1.1/4), S2 = C - ((H-L)*1.1/6)
    # where C = (H+L+C)/3 (typical price), H = high, L = low of prior completed bar
    
    # We need to calculate typical price for each completed 1h bar
    # Since we're working with 1h data, we can use the prior completed bar's OHLC
    typical_price = (high + low + close) / 3.0
    # Shift by 1 to use only prior completed 1h bar (no look-ahead)
    typical_price_shifted = np.roll(typical_price, 1)
    typical_price_shifted[0] = np.nan
    high_shifted = np.roll(high, 1)
    high_shifted[0] = np.nan
    low_shifted = np.roll(low, 1)
    low_shifted[0] = np.nan
    
    # Calculate Camarilla levels using prior completed bar's data
    hl_range = high_shifted - low_shifted
    camarilla_r3 = typical_price_shifted + (hl_range * 1.1 / 4.0)
    camarilla_s3 = typical_price_shifted - (hl_range * 1.1 / 4.0)
    
    # Align Camarilla levels to 1h timeframe (they're already at 1h, just need to shift for prior bar)
    camarilla_r3_aligned = camarilla_r3  # Already aligned to 1h, shifted for prior bar
    camarilla_s3_aligned = camarilla_s3  # Already aligned to 1h, shifted for prior bar
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + price > 4h EMA34 + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + price < 4h EMA34 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR price crosses below 4h EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR price crosses above 4h EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals