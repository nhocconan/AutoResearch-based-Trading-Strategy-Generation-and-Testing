#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and volume confirmation.
# Uses proven RSI mean reversion edge with HTF trend filter to avoid counter-trend trades.
# Long when RSI<30 and price>4h EMA50 (pullback in uptrend). Short when RSI>70 and price<4h EMA50 (bounce in downtrend).
# Volume confirmation (>1.3x 20-bar average) reduces false signals.
# Session filter (08-20 UTC) avoids low-liquidity periods.
# Position size 0.20 balances return and drawdown. Discrete levels minimize fee churn.
# Works in both bull and bear via 4h EMA50 trend filter.

name = "1h_RSI14_4hEMA50_Trend_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate volume confirmation: >1.3x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.3 * volume_ma_20
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma_20[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # RSI mean reversion conditions with volume confirmation
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        long_entry = rsi_oversold and price_above_ema and volume_confirm[i]
        short_entry = rsi_overbought and price_below_ema and volume_confirm[i]
        
        # Exit conditions: RSI returns to neutral zone (40-60) or trend reversal
        rsi_neutral = (rsi_values[i] >= 40) and (rsi_values[i] <= 60)
        long_exit = rsi_neutral or price_below_ema
        short_exit = rsi_neutral or price_above_ema
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals