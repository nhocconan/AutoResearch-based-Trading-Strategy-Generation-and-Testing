#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# - Camarilla levels (L3, L4, H3, H4) calculated from prior 1d OHLC
# - Long when price crosses above L3 with volume spike in choppy market (CHOP > 61.8)
# - Short when price crosses below H3 with volume spike in choppy market
# - Weekly trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - ATR(14) trailing stop (2.5x) on 12h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within HARD MAX: 200 total
# - Camarilla levels act as natural support/resistance in ranging markets
# - Volume spike confirms institutional interest at pivot levels
# - Chop filter ensures we only mean-revert in ranging markets, avoiding trends

name = "12h_1d_1w_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels from prior day OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 for completed bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute 1d Choppiness Index for regime filter
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
    
    # Choppiness Index = 100 * log10(sum(TR over n) / (n * max(HH-LL))) / log10(n)
    chop_period = 14
    tr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    hh = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    max_range = hh - ll
    chop = 100 * np.log10(tr_sum / (chop_period * max_range)) / np.log10(chop_period)
    chop = np.where(max_range == 0, 50, chop)  # handle division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 12h ATR for trailing stop
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = np.nan
    tr2_12h[0] = np.nan
    tr3_12h[0] = np.nan
    tr_12h = np.maximum.reduce([tr1_12h, tr2_12h, tr3_12h])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 12h volume and its 20-period moving average
    volume_12h = prices['volume'].values
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_12h[i]) or 
            np.isnan(volume_ma_20_12h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 12h data
        close_price = close_12h[i]
        high_price = high_12h[i]
        low_price = low_12h[i]
        volume_12h_current = volume_12h[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_spike = volume_12h_current > 1.5 * volume_ma_20_12h[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging market (mean revert)
        ranging_market = chop_aligned[i] > 61.8
        
        # Weekly trend filter
        weekly_uptrend = close_12h[i] > ema_50_aligned[i]
        weekly_downtrend = close_12h[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade in ranging markets with volume spike
            if ranging_market and volume_spike:
                # Long: price crosses above L3 (support) with weekly uptrend bias
                if close_price > camarilla_l3_aligned[i] and low_price <= camarilla_l3_aligned[i] and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: price crosses below H3 (resistance) with weekly downtrend bias
                elif close_price < camarilla_h3_aligned[i] and high_price >= camarilla_h3_aligned[i] and weekly_downtrend:
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
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_12h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_12h[i]
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