#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Donchian breakout: price breaks above/below 20-period high/low on 4h
# - Volume confirmation: current 4h volume > 1.3x 20-period average volume
# - Chop regime filter: only trade when Chop(14) < 38.2 (trending market) OR Chop(14) > 61.8 (ranging market)
# - In trending markets (Chop < 38.2): follow Donchian breakout direction
# - In ranging markets (Chop > 61.8): mean reversion at Donchian channels (fade extremes)
# - Weekly trend filter: avoid counter-trend trades using 1w EMA50
# - ATR(14) trailing stop (2.5x) on 4h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within HARD MAX: 400 total

name = "4h_1d_donchian_volume_chop_regime_v1"
timeframe = "4h"
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
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR for trailing stop
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 4h volume and its 20-period moving average
    volume_4h = prices['volume'].values
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d Chopiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = np.nan
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum.reduce([tr1_1d, tr2_1d, tr3_1d])
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index formula: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    hh_ll_diff = hh_14 - ll_14
    chop_raw = np.where(hh_ll_diff > 0, tr_sum_14 / hh_ll_diff, 1.0)
    chop_raw = np.where(chop_raw > 0, chop_raw, 1.0)
    chop_1d = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(volume_ma_20_4h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 4h volume for filter
        volume_4h_current = volume_4h[i]
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_spike = volume_4h_current > 1.3 * volume_ma_20_4h[i]
        
        # Chop regime filters
        chop_value = chop_aligned[i]
        trending_market = chop_value < 38.2  # Chop < 38.2 = trending
        ranging_market = chop_value > 61.8   # Chop > 61.8 = ranging
        
        # Weekly trend filter
        close_4h_current = close_4h[i]
        weekly_uptrend = close_4h_current > ema_50_aligned[i]
        weekly_downtrend = close_4h_current < ema_50_aligned[i]
        
        close_price = close_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Trending market: follow Donchian breakout
            if trending_market and volume_spike:
                # Long: price breaks above Donchian high AND weekly uptrend
                if close_price > donchian_high[i] and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: price breaks below Donchian low AND weekly downtrend
                elif close_price < donchian_low[i] and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            # Ranging market: mean reversion at Donchian channels
            elif ranging_market and volume_spike:
                # Long: price touches or goes below Donchian low (oversold) AND weekly uptrend bias
                if close_price <= donchian_low[i] and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: price touches or goes above Donchian high (overbought) AND weekly downtrend bias
                elif close_price >= donchian_high[i] and weekly_downtrend:
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