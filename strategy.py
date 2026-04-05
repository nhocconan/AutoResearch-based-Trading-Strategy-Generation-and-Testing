#!/usr/bin/env python3
"""
Experiment #8235: 6-hour Ichimoku Cloud Filter with Weekly Trend Filter.
Hypothesis: Ichimoku conversion/base line cross combined with 1-week trend filter captures 
trend continuation moves while avoiding false signals. The weekly filter ensures we only 
trade in the direction of the higher timeframe trend, reducing whipsaw in both bull and bear markets. 
Targeting 50-150 total trades over 4 years for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8235_6h_ichimoku1w_filter_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9   # Conversion line
KIJUN_PERIOD = 26   # Base line
SENKOU_B_PERIOD = 52 # Leading Span B
WEEKLY_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=WEEKLY_EMA_PERIOD, adjust=False, min_periods=WEEKLY_EMA_PERIOD).mean().values
    weekly_trend = np.where(close_1w > ema_1w, 1, -1)  # 1=uptrend, -1=downtrend
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_high = pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max()
    tenkan_low = pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_high = pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max()
    kijun_low = pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_high = pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max()
    senkou_b_low = pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()
    senkou_b = ((senkou_b_high + senkou_b_low) / 2)
    
    # Cloud (Kumo) is between Senkou Span A and B
    # For trend: price above cloud = bullish, below cloud = bearish
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, ATR_PERIOD) + 26
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_trend_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Get current Ichimoku values (no look-ahead: use current bar's values)
        # Note: Senkou spans are plotted 26 periods ahead, so we check current price vs cloud
        # that was calculated 26 periods ago
        if i >= 26:
            senkou_a_val = senkou_a.iloc[i-26] if not np.isnan(senkou_a.iloc[i-26]) else 0
            senkou_b_val = senkou_b.iloc[i-26] if not np.isnan(senkou_b.iloc[i-26]) else 0
        else:
            senkou_a_val = 0
            senkou_b_val = 0
            
        # Cloud boundaries (Senkou A and B form the cloud)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Tenkan/Kijun cross
        tk_cross = tenkan_sen.iloc[i] - kijun_sen.iloc[i]
        tk_cross_prev = tenkan_sen.iloc[i-1] - kijun_sen.iloc[i-1] if i > 0 else 0
        
        # Bullish cross: TK crosses above
        bullish_cross = (tk_cross > 0) and (tk_cross_prev <= 0)
        # Bearish cross: TK crosses below
        bearish_cross = (tk_cross < 0) and (tk_cross_prev >= 0)
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Weekly trend filter
        weekly_uptrend = weekly_trend_aligned[i] == 1
        weekly_downtrend = weekly_trend_aligned[i] == -1
        
        # Entry conditions
        long_entry = bullish_cross and price_above_cloud and weekly_uptrend
        short_entry = bearish_cross and price_below_cloud and weekly_downtrend
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals