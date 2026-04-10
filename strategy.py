#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and chop regime filter
# - Entry Long: Williams %R(14) < -80 (oversold) + 1d volume > 1.5x 20-period average + chop > 61.8 (rangy market)
# - Entry Short: Williams %R(14) > -20 (overbought) + 1d volume > 1.5x 20-period average + chop > 61.8
# - Exit: Williams %R crosses above -50 (for long) or below -50 (for short) OR ATR(21) trailing stop (2.5x)
# - Position sizing: 0.25 (discrete levels)
# - Uses 1d for volume confirmation and chop filter to avoid lower timeframe noise
# - Williams %R captures reversals in ranging markets, volume confirms participation
# - Target: 30-60 trades/year (120-240 total over 4 years) to stay within HARD MAX: 400 total

name = "4h_1d_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d OHLC for volume and chop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) on 4h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14)) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Calculate Chopiness Index(14) on 1d: 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    # ATR1 for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = np.nan
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum.reduce([tr1_1d, tr2_1d, tr3_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = np.where(
        (sum_atr_14 > 0) & (np.arange(len(sum_atr_14)) > 0),
        100 * np.log10(sum_atr_14 / (14 * np.log10(14))) / np.log10(14),
        50  # neutral when invalid
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 4h ATR(21) for trailing stop
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_4h = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    williams_r_prev = williams_r[0] if len(williams_r) > 0 else -50
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr_4h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            williams_r_prev = williams_r[i] if not np.isnan(williams_r[i]) else williams_r_prev
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period average
        volume_confirmation = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        # Chop filter: chop > 61.8 (rangy market)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + volume confirmation + chop filter
            if williams_r[i] < -80 and volume_confirmation and chop_filter:
                position = 1
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + volume confirmation + chop filter
            elif williams_r[i] > -20 and volume_confirmation and chop_filter:
                position = -1
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_4h[i]
                # Williams %R exit: crosses above -50
                williams_exit = williams_r[i] > -50 and williams_r_prev <= -50
                exit_condition = trailing_stop or williams_exit
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_4h[i]
                # Williams %R exit: crosses below -50
                williams_exit = williams_r[i] < -50 and williams_r_prev >= -50
                exit_condition = trailing_stop or williams_exit
            
            if exit_condition:
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        
        williams_r_prev = williams_r[i]
    
    return signals