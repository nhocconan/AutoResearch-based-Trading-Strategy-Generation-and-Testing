#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend direction with RSI extremes and chop regime filter
    # KAMA adapts to market noise - trending when ER high, ranging when ER low
    # RSI < 30 or > 70 for mean reentry in chop, RSI 40-60 for trend continuation
    # Chop filter: > 61.8 = range (fade extremes), < 38.2 = trend (follow KAMA)
    # Weekly HTF trend filter: only trade in direction of weekly KAMA
    # Target: 15-25 trades/year per symbol (60-100 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for KAMA, RSI, Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d KAMA (ER=10, fastest=2, slowest=30)
    def calculate_kama(close, er_period=10, fastest=2, slowest=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < er_period + 1:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'shape') else np.sum(np.abs(np.diff(close)))
        # Manual calculation for efficiency
        er = np.full(n, np.nan)
        for i in range(er_period, n):
            price_change = np.abs(close[i] - close[i-er_period])
            vol_sum = 0
            for j in range(i-er_period+1, i+1):
                vol_sum += np.abs(close[j] - close[j-1])
            if vol_sum > 0:
                er[i] = price_change / vol_sum
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = np.full(n, np.nan)
        fastest_sc = 2 / (fastest + 1)
        slowest_sc = 2 / (slowest + 1)
        for i in range(er_period, n):
            sc[i] = (er[i] * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # KAMA calculation
        kama[er_period] = close[er_period]
        for i in range(er_period + 1, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        
        return kama
    
    kama_1d = calculate_kama(close_1d, er_period=10, fastest=2, slowest=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1d RSI (14-period)
    def calculate_rsi(close, period=14):
        n = len(close)
        rsi = np.full(n, 50.0)
        if n < period + 1:
            return rsi
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # Initial average
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        # Calculate RSI
        for i in range(period, n):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Chop regime
    def calculate_chop(high, low, close, period=14):
        n = len(close)
        chop = np.full(n, 50.0)
        if n < period:
            return chop
        
        # True Range
        tr = np.full(n, np.nan)
        for i in range(n):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Sum of TR and max/min range
        sum_tr = np.full(n, np.nan)
        max_min_range = np.full(n, np.nan)
        
        for i in range(period-1, n):
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
            max_high = np.max(high[i-period+1:i+1])
            min_low = np.min(low[i-period+1:i+1])
            max_min_range[i] = max_high - min_low
            
            if max_min_range[i] > 0:
                chop[i] = 100 * np.log10(sum_tr[i] / max_min_range[i]) / np.log10(period)
        
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly KAMA for trend filter
    kama_1w = calculate_kama(close_1w, er_period=10, fastest=2, slowest=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime
        is_ranging = chop_1d_aligned[i] > 61.8
        is_trending = chop_1d_aligned[i] < 38.2
        
        # Weekly trend direction
        weekly_uptrend = close_1w[-1] > kama_1w_aligned[i] if len(close_1w) > 0 else False
        weekly_downtrend = close_1w[-1] < kama_1w_aligned[i] if len(close_1w) > 0 else False
        
        long_entry = False
        short_entry = False
        long_exit = False
        short_exit = False
        
        if is_ranging:
            # Mean reversion at RSI extremes
            long_entry = (rsi_1d_aligned[i] < 30) and weekly_uptrend
            short_entry = (rsi_1d_aligned[i] > 70) and weekly_downtrend
            long_exit = rsi_1d_aligned[i] > 50
            short_exit = rsi_1d_aligned[i] < 50
        elif is_trending:
            # Trend continuation with KAMA pullback
            long_entry = (close[i] > kama_1d_aligned[i]) and (rsi_1d_aligned[i] > 40) and (rsi_1d_aligned[i] < 60) and weekly_uptrend
            short_entry = (close[i] < kama_1d_aligned[i]) and (rsi_1d_aligned[i] > 40) and (rsi_1d_aligned[i] < 60) and weekly_downtrend
            long_exit = close[i] < kama_1d_aligned[i]
            short_exit = close[i] > kama_1d_aligned[i]
        else:
            # Neutral chop - no trades
            long_entry = False
            short_entry = False
            long_exit = True
            short_exit = True
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_kama_rsi_chop_weekly_filter_v1"
timeframe = "1d"
leverage = 1.0