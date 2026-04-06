#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Relative Strength Index (RSI) with Volume Confirmation.
# Uses daily 200-period EMA as trend filter to ensure alignment with higher timeframe trend.
# RSI(14) < 30 for long entries, > 70 for short entries, only in direction of daily trend.
# Volume filter (current volume > 1.3x 20-period average) ensures momentum confirmation.
# Designed to work in both bull and bear markets by filtering trades with daily trend.
# Target: 75-150 trades over 4 years (19-38/year).

name = "6s_rsi_trend_filter_v1"
timeframe = "6h"
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
    
    # Daily 200 EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(200) on daily close
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        alpha = 2.0 / (200 + 1)
        ema200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema200_1d[i-1]
    
    # Align daily EMA200 to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(14) calculation
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[np.isnan(rs)] = 100
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Trend filter: price above/below daily EMA200
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (rsi[i] >= 70 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (rsi[i] <= 30 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            if volume_filter:
                # Long: RSI < 30 (oversold) in uptrend
                if (rsi[i] < 30 and 
                    rsi[i-1] >= 30 and 
                    uptrend):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 70 (overbought) in downtrend
                elif (rsi[i] > 70 and 
                      rsi[i-1] <= 70 and 
                      downtrend):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Relative Strength Index (RSI) with Volume Confirmation.
# Uses daily 200-period EMA as trend filter to ensure alignment with higher timeframe trend.
# RSI(14) < 30 for long entries, > 70 for short entries, only in direction of daily trend.
# Volume filter (current volume > 1.3x 20-period average) ensures momentum confirmation.
# Designed to work in both bull and bear markets by filtering trades with daily trend.
# Target: 75-150 trades over 4 years (19-38/year).

name = "6s_rsi_trend_filter_v1"
timeframe = "6h"
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
    
    # Daily 200 EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(200) on daily close
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        alpha = 2.0 / (200 + 1)
        ema200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema200_1d[i-1]
    
    # Align daily EMA200 to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(14) calculation
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[np.isnan(rs)] = 100
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Trend filter: price above/below daily EMA200
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (rsi[i] >= 70 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (rsi[i] <= 30 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            if volume_filter:
                # Long: RSI < 30 (oversold) in uptrend
                if (rsi[i] < 30 and 
                    rsi[i-1] >= 30 and 
                    uptrend):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: RSI > 70 (overbought) in downtrend
                elif (rsi[i] > 70 and 
                      rsi[i-1] <= 70 and 
                      downtrend):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals