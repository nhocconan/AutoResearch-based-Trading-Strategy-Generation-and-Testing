#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and ADX trend filter
# - Long when price breaks above Camarilla H3 level with volume > 2.0x average AND ADX > 25
# - Short when price breaks below Camarilla L3 level with volume > 2.0x average AND ADX > 25
# - Exit when price retests Camarilla H4/L4 levels or volume drops below average
# - Uses 1d HTF for Camarilla pivot calculation (more stable than intraday)
# - Volume confirmation prevents false breakouts
# - ADX filter ensures trending market conditions
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Camarilla pivots work well in both trending and ranging markets when combined with volume and trend filters

name = "12h_1d_camarilla_pivot_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.25*(high-low)
    # H2 = close + 1.125*(high-low)
    # H1 = close + 1.075*(high-low)
    # L1 = close - 1.075*(high-low)
    # L2 = close - 1.125*(high-low)
    # L3 = close - 1.25*(high-low)
    # L4 = close - 1.5*(high-low)
    
    # Calculate daily range
    daily_range = df_1d['high'] - df_1d['low']
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # Calculate Camarilla levels for previous day
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_h2 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_h1 = prev_close + 1.075 * (prev_high - prev_low)
    camarilla_l1 = prev_close - 1.075 * (prev_high - prev_low)
    camarilla_l2 = prev_close - 1.125 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.25 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values, additional_delay_bars=1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values, additional_delay_bars=1)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2.values, additional_delay_bars=1)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1.values, additional_delay_bars=1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1.values, additional_delay_bars=1)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2.values, additional_delay_bars=1)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values, additional_delay_bars=1)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values, additional_delay_bars=1)
    
    # Pre-compute 12h ADX for trend filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed values
    tr_smooth = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # First smoothed value is simple average
    if len(tr) >= period:
        tr_smooth[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
    
    # Subsequent values using Wilder's smoothing
    for i in range(period + 1, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Smoothed DX (ADX)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= period:
        adx[period] = np.nansum(dx[1:period+1]) / period
        for i in range(period + 1, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after sufficient warmup for ADX
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 level with volume spike AND ADX > 25
            if (prices['high'].iloc[i] > h3_aligned[i] and 
                vol_spike.iloc[i] and 
                adx[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < L3 level with volume spike AND ADX > 25
            elif (prices['low'].iloc[i] < l3_aligned[i] and 
                  vol_spike.iloc[i] and 
                  adx[i] > 25):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests H4/L4 levels (strong reversal signal)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['low'].iloc[i] <= h4_aligned[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] >= l4_aligned[i] or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals