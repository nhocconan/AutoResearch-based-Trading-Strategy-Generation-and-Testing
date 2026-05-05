#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily RSI extreme reversal with 12h EMA34 trend filter and volume spike confirmation
# Long when RSI(14) crosses above 30 (oversold) AND price > 12h EMA34 AND volume > 1.8 * avg_volume(20) on 12h
# Short when RSI(14) crosses below 70 (overbought) AND price < 12h EMA34 AND volume > 1.8 * avg_volume(20) on 12h
# Exit when RSI crosses back through 50 (mean reversion midpoint) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# RSI provides timely reversal signals in ranging markets
# 12h EMA34 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms reversal strength and reduces false signals
# Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend)

name = "12h_RSI_EXT_12hEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least one completed daily bar for RSI
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily timeframe
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align RSI to 12h timeframe (wait for completed daily bar)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Get 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI crosses above 30 (oversold), above 12h EMA34, volume confirmation, in session
            if (rsi_aligned[i] > 30 and rsi_aligned[i-1] <= 30 and 
                close[i] > ema34_12h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 70 (overbought), below 12h EMA34, volume confirmation, in session
            elif (rsi_aligned[i] < 70 and rsi_aligned[i-1] >= 70 and 
                  close[i] < ema34_12h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses back above 50 (mean reversion) OR volume drops below average
            if rsi_aligned[i] > 50 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses back below 50 (mean reversion) OR volume drops below average
            if rsi_aligned[i] < 50 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals