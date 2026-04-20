#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Trend Following with Daily Pullback Entries
# Uses weekly trend filter (price above/below weekly 50 EMA) to establish direction
# On daily timeframe, enters on pullbacks to daily 20 EMA in direction of weekly trend
# Volume confirmation filters out false breakouts (volume > 1.5x 20-day average)
# ATR-based risk management with 2x ATR stop loss
# Designed for low frequency (10-25 trades/year) to minimize fee drag
# Works in both bull and bear markets by following the dominant weekly trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate 50-period EMA on weekly timeframe for trend filter
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Daily indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily 20 EMA for pullback entries
    ema20_daily = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_weekly_aligned[i]) or np.isnan(ema20_daily[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema50_weekly_aligned[i]
        weekly_downtrend = close[i] < ema50_weekly_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: weekly uptrend + price pulls back to/touches daily 20 EMA + volume
            long_signal = weekly_uptrend and (price <= ema20_daily[i] * 1.005) and has_volume
            
            # Short entry: weekly downtrend + price pulls back to/touches daily 20 EMA + volume
            short_signal = weekly_downtrend and (price >= ema20_daily[i] * 0.995) and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or weekly trend reversal
            stop_loss = entry_price - 2.0 * atr[i]
            trend_reversal = close[i] < ema50_weekly_aligned[i]
            
            if stop_loss <= 0 or price <= stop_loss or trend_reversal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or weekly trend reversal
            stop_loss = entry_price + 2.0 * atr[i]
            trend_reversal = close[i] > ema50_weekly_aligned[i]
            
            if stop_loss <= 0 or price >= stop_loss or trend_reversal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_DailyPullback_Volume_ATR"
timeframe = "1d"
leverage = 1.0