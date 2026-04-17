#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA21 trend filter and volume spike confirmation.
# Goes long when RSI < 30 (oversold) and price > 4h EMA21 (uptrend), short when RSI > 70 (overbought) and price < 4h EMA21 (downtrend).
# Uses 4h EMA21 for trend direction to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Volume spike (>1.5x 20-bar average) confirms momentum behind the move.
# Designed for 1h timeframe with tight entry conditions to limit trades to 15-35/year.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA21
    close_4h_series = pd.Series(close_4h)
    ema21_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h EMA21 to 1h
    ema21_1h = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need RSI(14) + EMA21(4h) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_1h[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to allow some trades)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend filter: price relative to 4h EMA21
        price_above_ema = close[i] > ema21_1h[i]
        price_below_ema = close[i] < ema21_1h[i]
        
        if position == 0:
            # Long: RSI oversold + price above 4h EMA21 + volume spike
            if (rsi_oversold and price_above_ema and volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought + price below 4h EMA21 + volume spike
            elif (rsi_overbought and price_below_ema and volume_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or price crosses below 4h EMA21
            if (rsi[i] >= 50) or (close[i] < ema21_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or price crosses above 4h EMA21
            if (rsi[i] <= 50) or (close[i] > ema21_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA21_VolumeMeanReversion"
timeframe = "1h"
leverage = 1.0