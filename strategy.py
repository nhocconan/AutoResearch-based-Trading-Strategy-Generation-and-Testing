#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d trend filter and session filter
# Uses 4h EMA(50) for trend direction and 1d ADX(14) for regime filter
# In trending regime (ADX > 25): trade in direction of 4h EMA(50)
# In ranging regime (ADX <= 25): fade Camarilla extremes
# Uses Camarilla pivot levels (H3/L3 for breakout, H4/L4 for fade) on 1h
# Session filter: 08-20 UTC to avoid low-volume periods
# Position size 0.20 to limit drawdown and enable discrete levels
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Works in both bull/bear: adapts to regime via ADX filter and uses multiple timeframes

name = "1h_4h_1d_camarilla_regime_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 4h data ONCE before loop for EMA(50) trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX(14) regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ADX(14) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    # Directional Movement
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full(len(data), np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Calculate smoothed values
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate DI and DX
    plus_di_14 = np.full(len(df_1d), np.nan)
    minus_di_14 = np.full(len(df_1d), np.nan)
    dx_14 = np.full(len(df_1d), np.nan)
    
    for i in range(14, len(df_1d)):
        if tr_14[i] != 0:
            plus_di_14[i] = (plus_dm_14[i] / tr_14[i]) * 100
            minus_di_14[i] = (minus_dm_14[i] / tr_14[i]) * 100
            if (plus_di_14[i] + minus_di_14[i]) != 0:
                dx_14[i] = (abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx_14 = wilders_smoothing(dx_14, 14)
    adx_14_1h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 1h Camarilla pivot levels (based on previous day)
    # We'll use rolling window of 24 periods (1d of 1h data) for pivot calculation
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(24, n):
        # Use previous 24 bars (1 day) for pivot calculation
        start_idx = i - 24
        end_idx = i
        if end_idx <= n:
            period_high = np.max(high[start_idx:end_idx])
            period_low = np.min(low[start_idx:end_idx])
            period_close = close[end_idx-1]  # Close of previous bar
            
            range_val = period_high - period_low
            if range_val > 0:
                camarilla_h3[i] = period_close + range_val * 1.1 / 4
                camarilla_l3[i] = period_close - range_val * 1.1 / 4
                camarilla_h4[i] = period_close + range_val * 1.1 / 2
                camarilla_l4[i] = period_close - range_val * 1.1 / 2
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(adx_14_1h[i]) or
            np.isnan(camarilla_h3[i]) or
            np.isnan(camarilla_l3[i]) or
            np.isnan(camarilla_h4[i]) or
            np.isnan(camarilla_l4[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        adx = adx_14_1h[i]
        ema_trend = ema_50_4h_aligned[i]
        curr_close = close[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when price crosses below 4h EMA(50)
                if curr_close < ema_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # Ranging regime
                # Exit when price returns to mean (crosses Camarilla H3)
                if curr_close > camarilla_h3[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending regime
                # Exit when price crosses above 4h EMA(50)
                if curr_close > ema_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            else:  # Ranging regime
                # Exit when price returns to mean (crosses Camarilla L3)
                if curr_close < camarilla_l3[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            # Entry logic based on regime
            if adx > 25:  # Trending regime - follow 4h EMA trend
                # Go long when price crosses above 4h EMA(50)
                # Go short when price crosses below 4h EMA(50)
                if i > 30:
                    prev_close = close[i-1]
                    if prev_close <= ema_50_4h_aligned[i-1] and curr_close > ema_trend:
                        position = 1
                        signals[i] = 0.20
                    elif prev_close >= ema_50_4h_aligned[i-1] and curr_close < ema_trend:
                        position = -1
                        signals[i] = -0.20
            else:  # Ranging regime - fade Camarilla extremes
                # Go long when price crosses below Camarilla L4 (oversold)
                # Go short when price crosses above Camarilla H4 (overbought)
                if i > 30:
                    prev_close = close[i-1]
                    if prev_close >= camarilla_l4[i-1] and curr_close < camarilla_l4[i]:
                        position = 1
                        signals[i] = 0.20
                    elif prev_close <= camarilla_h4[i-1] and curr_close > camarilla_h4[i]:
                        position = -1
                        signals[i] = -0.20
    
    return signals