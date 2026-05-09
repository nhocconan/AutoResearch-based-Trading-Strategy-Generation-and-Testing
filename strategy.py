#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily RSI reversal with weekly trend filter and volume confirmation
# RSI < 30 (oversold) + weekly close > weekly EMA20 (bullish trend) + volume spike -> long
# RSI > 70 (overbought) + weekly close < weekly EMA20 (bearish trend) + volume spike -> short
# Uses daily timeframe with weekly trend filter to avoid counter-trend trades
# Designed to work in both bull and bear markets by following higher timeframe trend

name = "Daily_RSI_Reversal_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend
    close_1w = pd.Series(df_1w['close'].values)
    ema_20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily RSI(14)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral RSI when no loss
    
    # Daily volume spike detection (20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        price = close[i]
        weekly_trend_up = price > ema_20_1w_aligned[i]  # Price above weekly EMA20
        weekly_trend_down = price < ema_20_1w_aligned[i]  # Price below weekly EMA20
        
        if position == 0:
            # Long: Oversold + weekly uptrend + volume
            if rsi[i] < 30 and weekly_trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + weekly downtrend + volume
            elif rsi[i] > 70 and weekly_trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or trend change
            if rsi[i] > 70 or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or trend change
            if rsi[i] < 30 or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals