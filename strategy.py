#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA crossover for trend direction and 1h RSI extremes for mean reversion entries
# Long when 4h trend is bullish (EMA20 > EMA50) AND 1h RSI < 30 (oversold) AND volume > 1.2x 20-period average AND within active session (08-20 UTC)
# Short when 4h trend is bearish (EMA20 < EMA50) AND 1h RSI > 70 (overbought) AND volume > 1.2x 20-period average AND within active session
# Exits on RSI reversal to 50 or 4h trend change. Uses volume confirmation and session filter to reduce noise trades.
# Target: 15-37 trades/year on 1h by combining 4h trend filter with 1h mean reversion entries.
# Works in bull markets via buying dips in uptrends and bear markets via selling rallies in downtrends.

name = "1h_4hEMA_Cross_RSI_MeanReversion_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 and EMA50 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_4h = ema_20_4h > ema_50_4h
    trend_bearish_4h = ema_20_4h < ema_50_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish_4h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish_4h.astype(float))
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.2)  # Volume at least 1.2x average for confirmation
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade during active session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: 4h bullish trend AND 1h RSI < 30 (oversold) AND volume spike
            if (trend_bullish_aligned[i] > 0.5 and 
                rsi_values[i] < 30 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: 4h bearish trend AND 1h RSI > 70 (overbought) AND volume spike
            elif (trend_bearish_aligned[i] > 0.5 and 
                  rsi_values[i] > 70 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) OR 4h trend turns bearish
            if (rsi_values[i] > 50 or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) OR 4h trend turns bullish
            if (rsi_values[i] < 50 or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals