#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + 1d Camarilla Pivot Breakout with Volume Confirmation
# ADX > 25 identifies trending markets on 6h. Enter breakouts at 1d Camarilla R4/S4 levels
# with volume spike and ADX confirmation. Designed for 50-150 total trades over 4 years
# (12-37/year) to minimize fee drag. Works in bull markets via upside breakouts and
# in bear markets via downside breakdowns with trend filter.

name = "6h_ADX_Camarilla_R4S4_Breakout_1dVolumeSpike"
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r4 = typical_price + (hl_range * 1.1 / 2)
    camarilla_s4 = typical_price - (hl_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Calculate ADX on 6h data (14-period)
    def calculate_atr(high, low, close, length):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr = pd.Series(tr).ewm(span=length, adjust=False, min_periods=length).mean().values
        return atr
    
    def calculate_dm(high, low):
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        return plus_dm, minus_dm
    
    if len(close) >= 14:
        atr = calculate_atr(high, low, close, 14)
        plus_dm, minus_dm = calculate_dm(high, low)
        
        # Avoid division by zero
        atr_safe = np.where(atr == 0, 1e-10, atr)
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_safe
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_safe
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx = np.full(n, np.nan)
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start from 14 to have valid ADX values
        # Skip if any value is NaN or outside session
        if (np.isnan(adx[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 with ADX > 25 and volume spike
            if close[i] > camarilla_r4_aligned[i] and adx[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with ADX > 25 and volume spike
            elif close[i] < camarilla_s4_aligned[i] and adx[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below R3 or ADX weakens
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, (typical_price + (hl_range * 1.1/4)).values)
            if close[i] < camarilla_r3_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above S3 or ADX weakens
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, (typical_price - (hl_range * 1.1/4)).values)
            if close[i] > camarilla_s3_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals