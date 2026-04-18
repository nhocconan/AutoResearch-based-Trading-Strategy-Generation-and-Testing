# ANALYSIS
# The key failure pattern: overtrading (>1000 trades/year) due to loose entry conditions.
# From experiment history: 4h_Donchian20_1dEMA34_RSI_VolumeFilter_v1 had 1394 trades/sym.
# To reduce trades: make entry conditions stricter by requiring multiple confirmations.
# Hypothesis: Combine Donchian breakout with volume spike AND RSI reversal (not just non-extreme)
# for higher quality signals. Use 1d timeframe for Donchian/EMA/RSI as in the original,
# but add stricter filters to reduce trade frequency while maintaining edge.
# Target: 20-50 trades/year on 4h (80-200 total over 4 years) to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 34-period EMA on daily for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 14-day RSI for momentum and reversal signals
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all daily data to 4h timeframe
    upper_channel_4h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_4h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume spike: volume > 2.0x 20-period average (stricter than 1.5x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # RSI reversal signals: RSI < 30 for long entry, RSI > 70 for short entry
    rsi_oversold = rsi_4h < 30
    rsi_overbought = rsi_4h > 70
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_4h[i]) or np.isnan(lower_channel_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(rsi_4h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_34_4h[i]
        downtrend = close[i] < ema_34_4h[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + uptrend + volume spike + RSI oversold (reversal signal)
            if (close[i] > upper_channel_4h[i] and 
                uptrend and 
                volume_spike[i] and 
                rsi_oversold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + downtrend + volume spike + RSI overbought (reversal signal)
            elif (close[i] < lower_channel_4h[i] and 
                  downtrend and 
                  volume_spike[i] and 
                  rsi_overbought[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian OR trend reverses OR RSI overbought
            if (close[i] < lower_channel_4h[i]) or (not uptrend) or (rsi_overbought[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian OR trend reverses OR RSI oversold
            if (close[i] > upper_channel_4h[i]) or (not downtrend) or (rsi_oversold[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_RSIReversal_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0