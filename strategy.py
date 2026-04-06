#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and 1d volume confirmation.
# RSI < 30 oversold/ > 70 overbought for entries.
# 4h EMA(50) trend filter: only long when price > EMA50, short when price < EMA50.
# 1d volume spike (>1.5x 20-period average) confirms institutional participation.
# Designed for 1h timeframe targeting 60-150 trades over 4 years with session filter (08-20 UTC).

name = "1h_rsi4hema1dvol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14]) if len(gain) >= 14 else np.nan
    avg_loss[13] = np.mean(loss[1:14]) if len(loss) >= 14 else np.nan
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.zeros_like(close_4h)
    ema_4h[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        ema_4h[i] = (close_4h[i] * 2 / (50 + 1)) + (ema_4h[i-1] * (48 / (50 + 1)))
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.zeros_like(volume_1d)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(14, 19)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Trend condition from 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI overbought or stoploss
            if (rsi[i] > 70 or 
                close[i] < entry_price - 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI oversold or stoploss
            if (rsi[i] < 30 or 
                close[i] > entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume filter and session filter (08-20 UTC)
            hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
            in_session = 8 <= hour <= 20
            
            if volume_filter and in_session:
                # Long: oversold in uptrend
                if rsi[i] < 30 and uptrend:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: overbought in downtrend
                elif rsi[i] > 70 and downtrend:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals