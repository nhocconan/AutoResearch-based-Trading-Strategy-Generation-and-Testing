#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and 1d volume confirmation
# Long when RSI(2) < 10, price > 4h EMA50, and 1d volume > 1.5x 20-day average
# Short when RSI(2) > 90, price < 4h EMA50, and 1d volume > 1.5x 20-day average
# Uses 4h EMA for trend direction (avoid counter-trend trades), 1d volume spike for conviction
# RSI(2) captures extreme short-term reversals in both bull and bear markets
# Designed for low trade frequency (15-37/year on 1h) to minimize fee drag
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets

name = "1h_RSI2_4hEMA50_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe (wait for completed 4h bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d for confirmation
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 1h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate RSI(2) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 2, 20)  # EMA(50), RSI(2), volume MA(20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: RSI(2) < 10, price > 4h EMA50, volume spike
            if (rsi_values[i] < 10 and close[i] > ema_50_4h_aligned[i] and 
                volume[i] > (vol_ma_20_1d_aligned[i] * 1.5)):
                signals[i] = 0.20
                position = 1
            # Short entry: RSI(2) > 90, price < 4h EMA50, volume spike
            elif (rsi_values[i] > 90 and close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > (vol_ma_20_1d_aligned[i] * 1.5)):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: RSI(2) > 50 (mean reversion complete) or price < 4h EMA50 (trend break)
            if (rsi_values[i] > 50 or close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: RSI(2) < 50 (mean reversion complete) or price > 4h EMA50 (trend break)
            if (rsi_values[i] < 50 or close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals