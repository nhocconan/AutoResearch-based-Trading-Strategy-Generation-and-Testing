#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume spike and 1d choppiness regime filter
# - Camarilla levels (H4, L4) from prior 1d session act as intraday support/resistance
# - Breakout above H4 with volume spike = long entry; breakdown below L4 with volume spike = short
# - 1d Choppiness Index (CHOP) > 61.8 = ranging (avoid breakout trades), CHOP < 38.2 = trending (favor breakouts)
# - 12h volume confirmation: current 4h volume > 1.8x 20-period average reduces false breakouts
# - ATR(14) trailing stop (2.5x) on 4h timeframe for risk management
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) within HARD MAX: 400 total
# - Works in both bull/bear: breakouts capture momentum, chop filter avoids whipsaws in ranges

name = "4h_12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 50 or len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on prior day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Pre-compute 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR14 over last 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over last 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    mask = (sum_atr_14 > 0) & (range_14 > 0) & (~np.isnan(sum_atr_14)) & (~np.isnan(range_14))
    chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    chop = np.where(chop < 0, 0, chop)  # CHOP cannot be negative
    chop = np.where(chop > 100, 100, chop)  # Cap at 100
    
    # Pre-compute 12h volume and its 20-period moving average
    volume_12h = df_12h['volume'].values
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Pre-compute 4h ATR for trailing stop
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr1_4h[0] = np.nan
    tr2_4h[0] = np.nan
    tr3_4h[0] = np.nan
    tr_4h = np.maximum.reduce([tr1_4h, tr2_4h, tr3_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 4h volume and its 20-period moving average
    volume_4h = prices['volume'].values
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(volume_ma_20_4h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 4h data
        high_price = high_4h[i]
        low_price = low_4h[i]
        close_price = close_4h[i]
        volume_4h_current = volume_4h[i]
        
        # Volume confirmation: current 4h volume > 1.8x 20-period average
        volume_spike = volume_4h_current > 1.8 * volume_ma_20_4h[i]
        
        # Choppiness regime filter: CHOP < 38.2 = trending (favor breakouts), CHOP > 61.8 = ranging (avoid)
        trending_market = chop_aligned[i] < 38.2
        ranging_market = chop_aligned[i] > 61.8
        
        close_price = close_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade breakouts in trending markets
            if trending_market and volume_spike:
                # Long: break above H4 camarilla level
                if high_price > camarilla_h4_aligned[i]:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: break below L4 camarilla level
                elif low_price < camarilla_l4_aligned[i]:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_4h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals