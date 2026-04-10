#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d chop regime filter
# - Donchian breakout provides clear entry/exit signals with proven edge
# - Volume confirmation ensures breakouts have conviction
# - Choppiness index regime filter avoids false breakouts in sideways markets
# - Weekly trend filter aligns with higher timeframe momentum
# - ATR-based trailing stop manages risk
# - Discrete position sizing (0.25) minimizes fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) within HARD MAX: 200

name = "12h_1d_donchian_volume_chop_v1"
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
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper/lower bands
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(TR14) / (sum(|HH-LL|14) + sum(|LL-HH|14))) / log10(14)
    # Simplified: CHOP = 100 * log10(ATR14 / (HHV14 - LLV14)) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = hh_1d - ll_1d
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid division by zero
    chop_ratio = atr_1d / chop_denominator
    chop_ratio = np.where(np.isnan(chop_ratio) | (chop_ratio <= 0), 1e-10, chop_ratio)
    chop_1d = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
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
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
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
        
        # Chop regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        chop_value = chop_aligned[i]
        ranging_market = chop_value > 61.8
        trending_market = chop_value < 38.2
        
        # Weekly trend filter
        weekly_uptrend = close_12h[i] > ema_50_aligned[i]
        weekly_downtrend = close_12h[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Trending market: Donchian breakouts with volume
            if trending_market and volume_spike:
                # Long: price breaks above Donchian upper band AND weekly uptrend
                if close_price > donchian_upper_aligned[i] and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = high_price
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band AND weekly downtrend
                elif close_price < donchian_lower_aligned[i] and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = low_price
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            # Ranging market: mean reversion at Donchian extremes
            elif ranging_market:
                # Long: price touches Donchian lower band AND weekly uptrend bias
                if low_price <= donchian_lower_aligned[i] * 1.001 and weekly_uptrend:  # small buffer for touch
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = high_price
                    signals[i] = 0.25
                # Short: price touches Donchian upper band AND weekly downtrend bias
                elif high_price >= donchian_upper_aligned[i] * 0.999 and weekly_downtrend:  # small buffer for touch
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = low_price
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high_price)
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = close_price < highest_since_entry - 2.5 * atr_12h[i]
                # Also exit if price re-enters Donchian channel (mean reversion in ranging)
                reentry_exit = ranging_market and close_price < donchian_upper_aligned[i]
                exit_condition = trailing_stop or reentry_exit
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, low_price)
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = close_price > lowest_since_entry + 2.5 * atr_12h[i]
                # Also exit if price re-enters Donchian channel (mean reversion in ranging)
                reentry_exit = ranging_market and close_price > donchian_lower_aligned[i]
                exit_condition = trailing_stop or reentry_exit
            
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