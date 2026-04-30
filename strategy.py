#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter.
# Uses Camarilla pivot levels (R3, S3) from 1d data, requiring price to close beyond these levels
# with volume > 2.0x 20-bar average for confirmation. Only trades when 4h choppiness index
# is between 38.2 and 61.8 (avoiding extreme chop and strong trends). Discrete position sizing
# at ±0.25. ATR(14) trailing stop at 2.0x for risk management. Designed to work in both bull
# and bear markets by using volatility-based pivots and regime filtering to avoid whipsaws.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R3S3_1dVolumeSpike_ChopFilter_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels: R3, S3
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # We use R3 and S3 as key levels
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + (1.1 * hl_range * 1.1 / 4)
    camarilla_s3 = df_1d['close'] - (1.1 * hl_range * 1.1 / 4)
    r3_values = camarilla_r3.values
    s3_values = camarilla_s3.values
    
    # Align Camarilla levels to primary timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_values)
    
    # 1d volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_confirm = df_1d['volume'].values > (2.0 * vol_ma_20)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    # 4h choppiness index: CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high)-min(low)))) / log10(n)
    # We use a simplified version: CHOP = 100 * log10(ATR_sum / (n * range)) / log10(n)
    # Where CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want 38.2 <= CHOP <= 61.8 (avoiding extremes)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate 4h choppiness over 14 periods
    n_chop = 14
    atr_sum = pd.Series(atr).rolling(window=n_chop, min_periods=n_chop).sum().values
    max_high = pd.Series(high).rolling(window=n_chop, min_periods=n_chop).max().values
    min_low = pd.Series(low).rolling(window=n_chop, min_periods=n_chop).min().values
    range_n = max_high - min_low
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_sum / (n_chop * range_n + 1e-10)) / np.log10(n_chop)
    # Handle invalid values
    chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50.0, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(n_chop, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_raw[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_volume_confirm = volume_confirm_aligned[i]
        curr_atr = atr[i]
        curr_chop = chop_raw[i]
        
        # Chop regime filter: only trade when 38.2 <= CHOP <= 61.8
        chop_filter = (curr_chop >= 38.2) & (curr_chop <= 61.8)
        
        if position == 0:  # Flat - look for new entries
            # Long: price closes above R3, volume confirmation, chop filter
            if (curr_close > curr_r3 and 
                curr_volume_confirm and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price closes below S3, volume confirmation, chop filter
            elif (curr_close < curr_s3 and 
                  curr_volume_confirm and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.0*ATR from highest point
            if curr_close < highest_since_entry - (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest point
            if curr_close > lowest_since_entry + (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals