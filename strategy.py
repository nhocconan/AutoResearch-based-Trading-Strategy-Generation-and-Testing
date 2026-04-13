#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h primary with 1d/1w HTF - 1d Camarilla pivot breakout with volume confirmation and chop filter
    # Designed to capture institutional breakouts at key pivot levels with volume surge in trending markets
    # Target: 50-150 total trades over 4 years (12-37/year) for optimal fee drag and generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low), L3 = Close - 1.125*(High-Low)
    # H2 = Close + 0.75*(High-Low), L2 = Close - 0.75*(High-Low)
    # H1 = Close + 0.5*(High-Low), L1 = Close - 0.5*(High-Low)
    # Pivot = (High + Low + Close)/3
    
    # Use previous day's OHLC for today's pivots (no look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # First bar will have NaN due to roll, that's correct
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Key breakout levels: H3 and L3 (strong breakout levels)
    h3 = pivot + 1.125 * range_1d
    l3 = pivot - 1.125 * range_1d
    h4 = pivot + 1.5 * range_1d  # stronger breakout
    l4 = pivot - 1.5 * range_1d
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1d ATR (14-period) for volatility/chop filter
    def calculate_atr(high, low, close, window=14):
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        return pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, window=14)
    atr_ma_10 = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) for regime filter
    def calculate_chop(high, low, close, window=14):
        # True Range
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        
        # Sum of TR over window
        sum_tr = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Choppiness Index: 100 * log10(sum_tr / (highest_high - lowest_low)) / log10(window)
        # Avoid division by zero
        hh_ll = highest_high - lowest_low
        chop = np.where((hh_ll > 0) & ~np.isnan(sum_tr), 
                        100 * np.log10(sum_tr / hh_ll) / np.log10(window), 
                        50)  # default to neutral when invalid
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    
    # Align all HTF indicators to 12h primary timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    atr_ma_10_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_10)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(atr_ma_10_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-day average
        volume_confirmed = volume_1d[i] > 1.8 * vol_avg_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_1d[i] > 0.4 * atr_ma_10_aligned[i]
        
        # Regime filter: Choppiness Index < 61.8 (trending market) 
        # CHOP > 61.8 = ranging/choppy, CHOP < 38.2 = strong trend
        regime_filter = chop_1d_aligned[i] < 61.8
        
        # Trend filter: price above/below 1w EMA50
        # For longs: price > EMA50 (uptrend bias)
        # For shorts: price < EMA50 (downtrend bias)
        close_12h = close[i]  # current 12h close
        trend_filter_long = close_12h > ema_50_1w_aligned[i]
        trend_filter_short = close_12h < ema_50_1w_aligned[i]
        
        # Breakout conditions at Camarilla H3/L3 levels
        breakout_up = close_12h > h3_aligned[i]
        breakout_down = close_12h < l3_aligned[i]
        
        # Strong breakout conditions at H4/L4 (require less confirmation)
        strong_breakout_up = close_12h > h4_aligned[i]
        strong_breakout_down = close_12h < l4_aligned[i]
        
        # Entry conditions
        enter_long = (breakout_up or strong_breakout_up) and volume_confirmed and vol_filter and regime_filter and trend_filter_long
        enter_short = (breakout_down or strong_breakout_down) and volume_confirmed and vol_filter and regime_filter and trend_filter_short
        
        # Exit conditions: return to pivot level or opposite H3/L3
        exit_long = position == 1 and (close_12h <= pivot_aligned[i] or close_12h < l3_aligned[i])
        exit_short = position == -1 and (close_12h >= pivot_aligned[i] or close_12h > h3_aligned[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0