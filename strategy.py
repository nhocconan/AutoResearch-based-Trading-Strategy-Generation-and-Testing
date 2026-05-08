#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_Reversion_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-calculate hour filter
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for trend filter (Camarilla levels based on prior day)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate daily high/low/close from 4h data for Camarilla
    # Resample 4h to daily using proper method - but we'll use 1d data directly
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla levels: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # Where H,L,C are previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 4h trend filter: EMA crossover
    ema_fast = pd.Series(df_4h['close']).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_slow)
    
    # 1h RSI for mean reversion entry
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion long: price touches/slightly breaks L4, RSI oversold, 4h uptrend, volume confirmation
            long_cond = (close[i] <= camarilla_l4_aligned[i] * 1.002 and  # Allow small penetration
                        rsi[i] < 30 and
                        ema_fast_aligned[i] > ema_slow_aligned[i] and
                        volume_filter[i])
            
            # Mean reversion short: price touches/slightly breaks H4, RSI overbought, 4h downtrend, volume confirmation
            short_cond = (close[i] >= camarilla_h4_aligned[i] * 0.998 and  # Allow small penetration
                         rsi[i] > 70 and
                         ema_fast_aligned[i] < ema_slow_aligned[i] and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reaches midpoint or RSI overbought
            midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
            if close[i] >= midpoint or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reaches midpoint or RSI oversold
            midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
            if close[i] <= midpoint or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals