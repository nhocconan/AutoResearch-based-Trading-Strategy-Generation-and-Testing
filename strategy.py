#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Trend Filter with Daily Pullback Entry
# Uses 1-week EMA20 as trend filter - long when price above weekly EMA20, short when below
# Entry on daily pullback to EMA50 with RSI confirmation (RSI < 40 for long, > 60 for short)
# Volume confirmation requires volume > 1.3x 20-day average
# ATR-based stop loss manages risk (2x ATR)
# Designed for low-frequency trading (target: 15-25 trades/year) to minimize fee drag
# Weekly trend filter reduces whipsaws in sideways markets
# Pullback to EMA50 provides good risk-reward entries in trending markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly timeframe for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for pullback entries
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14) for entry confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: volume > 1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
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
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(ema50[i]) or \
           np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema20_1w_aligned[i]
        is_downtrend = close[i] < ema20_1w_aligned[i]
        
        # Price relative to daily EMA50
        near_ema50_long = abs(close[i] - ema50[i]) / ema50[i] < 0.02  # Within 2% of EMA50
        near_ema50_short = abs(close[i] - ema50[i]) / ema50[i] < 0.02  # Within 2% of EMA50
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: weekly uptrend + pullback to EMA50 + RSI oversold + volume
            long_signal = is_uptrend and near_ema50_long and rsi_oversold and has_volume
            
            # Short entry: weekly downtrend + pullback to EMA50 + RSI overbought + volume
            short_signal = is_downtrend and near_ema50_short and rsi_overbought and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or weekly trend change
            stop_loss = entry_price - 2.0 * atr[i]
            trend_change = close[i] < ema20_1w_aligned[i]  # Weekly trend turns down
            
            if stop_loss <= 0 or price <= stop_loss or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or weekly trend change
            stop_loss = entry_price + 2.0 * atr[i]
            trend_change = close[i] > ema20_1w_aligned[i]  # Weekly trend turns up
            
            if stop_loss <= 0 or price >= stop_loss or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_DailyPullback_RSI_Volume"
timeframe = "1d"
leverage = 1.0