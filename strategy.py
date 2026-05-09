#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Choppiness Index regime filter to identify trending vs ranging markets.
# In trending markets (CHOP < 38.2), trade Donchian(20) breakouts in the direction of 1d EMA50 trend.
# In ranging markets (CHOP > 61.8), trade mean-reversion at Bollinger Bands (20,2) with RSI(14) extremes.
# Volume confirmation required for all entries. Designed to work in both bull and bear markets by adapting to regime.
# Target: 20-30 trades per year to minimize fee drag and improve generalization.
name = "4h_RegimeAdaptive_Donchian_BB_MeanRev"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filter and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log(sum_tr / (atr_14 * 14)) / log(14)
    choppiness = 100 * np.log(sum_tr / (atr_14 * 14)) / np.log(14)
    choppiness_4h = align_htf_to_ltf(prices, df_1d, choppiness)
    
    # Donchian(20) channels on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Bollinger Bands (20,2) on 4h
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # RSI(14) on 4h
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: spike above 2.0x 24-period average (1 day of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 24)  # Wait for EMA50, Donchian, BB, RSI, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(choppiness_4h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Trending market: CHOP < 38.2
            if choppiness_4h[i] < 38.2:
                # Long: Donchian breakout above, uptrend (close > EMA50), volume
                if (close[i] > highest_20[i] and 
                    close[i] > ema_50_4h[i] and 
                    vol_ok and 
                    in_session):
                    signals[i] = 0.25
                    position = 1
                # Short: Donchian breakdown below, downtrend (close < EMA50), volume
                elif (close[i] < lowest_20[i] and 
                      close[i] < ema_50_4h[i] and 
                      vol_ok and 
                      in_session):
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP > 61.8
            elif choppiness_4h[i] > 61.8:
                # Long: mean reversion at lower BB, RSI oversold, volume
                if (close[i] < bb_lower[i] and 
                    rsi[i] < 30 and 
                    vol_ok and 
                    in_session):
                    signals[i] = 0.25
                    position = 1
                # Short: mean reversion at upper BB, RSI overbought, volume
                elif (close[i] > bb_upper[i] and 
                      rsi[i] > 70 and 
                      vol_ok and 
                      in_session):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            if choppiness_4h[i] < 38.2:  # trending market
                # Exit long: Donchian breakdown or trend reversal
                if close[i] < lowest_20[i] or close[i] < ema_50_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging market or transition
                # Exit long: price at middle BB or RSI overbought
                if close[i] > bb_middle[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            if choppiness_4h[i] < 38.2:  # trending market
                # Exit short: Donchian breakout or trend reversal
                if close[i] > highest_20[i] or close[i] > ema_50_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging market or transition
                # Exit short: price at middle BB or RSI oversold
                if close[i] < bb_middle[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals