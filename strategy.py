#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h/1d trend filter and volume spike confirmation
# In 4h uptrend (price > 4h EMA50), look for RSI < 30 (oversold) with volume > 2x average for longs
# In 4h downtrend (price < 4h EMA50), look for RSI > 70 (overbought) with volume > 2x average for shorts
# 1d EMA200 acts as higher timeframe regime filter: only trade long when price > 1d EMA200, short when price < 1d EMA200
# Uses strict volume confirmation (>2.0x 20-period average) and session filter (08-20 UTC) to reduce noise
# Target: 15-37 trades/year on 1h timeframe to minimize fee drag while capturing mean reversion in trends

name = "1h_RSI_MeanReversion_4hEMA50_1dEMA200_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA200 regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA200 for regime filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate RSI(14) on 1h data
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # volume MA and RSI warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi_values[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_ema200_1d = ema_200_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or stoploss via opposite signal
            if curr_rsi > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or stoploss via opposite signal
            if curr_rsi < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry when:
            # 1. 4h uptrend (price > 4h EMA50)
            # 2. 1d regime allows long (price > 1d EMA200)
            # 3. RSI < 30 (oversold)
            # 4. Volume confirmation
            if (curr_close > curr_ema50_4h and 
                curr_close > curr_ema200_1d and 
                curr_rsi < 30 and 
                vol_confirm):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short entry when:
            # 1. 4h downtrend (price < 4h EMA50)
            # 2. 1d regime allows short (price < 1d EMA200)
            # 3. RSI > 70 (overbought)
            # 4. Volume confirmation
            elif (curr_close < curr_ema50_4h and 
                  curr_close < curr_ema200_1d and 
                  curr_rsi > 70 and 
                  vol_confirm):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals