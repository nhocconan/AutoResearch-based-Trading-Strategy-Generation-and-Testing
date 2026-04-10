#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume spike confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (1d timeframe)
# - Regime filter: ADX(14) > 25 on 1d for trending markets, < 20 for ranging
# - Volume confirmation: current 6h volume > 1.5x 20-period average
# - Entry logic: 
#   * Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong uptrend) AND volume spike
#   * Short: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 (strong downtrend) AND volume spike
#   * In ranging markets (ADX < 20): mean reversion at extreme Bull/Bear Power
# - Weekly trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - ATR(14) trailing stop (2.0x) on 6h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Elder Ray effectively measures bull/bear strength relative to trend
# - ADX regime filter prevents whipsaws in sideways markets
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1w_elderray_volume_regime_v1"
timeframe = "6h"
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
    
    # Pre-compute 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    
    # Pre-compute 1d ADX for regime filter
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
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    adx = np.where((di_plus + di_minus) == 0, 0, adx)  # avoid division by zero
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 6h ATR for trailing stop
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = np.nan
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.maximum.reduce([tr1_6h, tr2_6h, tr3_6h])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h volume and its 20-period moving average
    volume_6h = prices['volume'].values
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(volume_ma_20_6h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 6h volume for filter
        volume_6h_current = volume_6h[i]
        
        # Get current 1d close for weekly trend filter (use raw close, aligned)
        close_1d_current = close_1d
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        
        # Elder Ray conditions
        bull_power_positive = bull_power_aligned[i] > 0
        bear_power_negative = bear_power_aligned[i] < 0
        
        # ADX regime filter
        strong_trend = adx_aligned[i] > 25
        ranging_market = adx_aligned[i] < 20
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_spike = volume_6h_current > 1.5 * volume_ma_20_6h[i]
        
        # Weekly trend filter
        weekly_uptrend = close_1d_aligned[i] > ema_50_aligned[i]
        weekly_downtrend = close_1d_aligned[i] < ema_50_aligned[i]
        
        close_price = close_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Trending market entries (ADX > 25)
            if strong_trend and volume_spike:
                # Long: Bull Power > 0 AND Bear Power < 0 AND weekly uptrend
                if bull_power_positive and bear_power_negative and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: Bear Power < 0 AND Bull Power > 0 AND weekly downtrend
                elif bear_power_negative and bull_power_positive and weekly_downtrend:
                    position = -1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    lowest_since_entry = prices['low'].iloc[i]
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            # Ranging market mean reversion (ADX < 20)
            elif ranging_market:
                # Long: Bear Power extremely negative (oversold) AND weekly uptrend bias
                if bear_power_aligned[i] < -np.std(bear_power[max(0, i-50):i]) * 2 and weekly_uptrend:
                    position = 1
                    entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                    highest_since_entry = prices['high'].iloc[i]
                    signals[i] = 0.25
                # Short: Bull Power extremely high (overbought) AND weekly downtrend bias
                elif bull_power_aligned[i] > np.std(bull_power[max(0, i-50):i]) * 2 and weekly_downtrend:
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
                # ATR trailing stop: exit when price drops 2.0*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.0 * atr_6h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.0*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.0 * atr_6h[i]
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