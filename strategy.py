#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion in 4h trend using Bollinger Bands and RSI
# Long when: 4h trend is up (close > EMA20_4h), price touches lower BB (1h), RSI < 30, volume > 1.5x avg
# Short when: 4h trend is down (close < EMA20_4h), price touches upper BB (1h), RSI > 70, volume > 1.5x avg
# Exit when RSI crosses 50 in opposite direction
# Uses 4h for trend direction, 1h for entry timing and mean reversion signals
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend direction
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 20-period EMA on 4h for trend
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Bollinger Bands on 1h (20-period, 2 std)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Calculate RSI on 1h (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 40
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        sma_val = sma_20[i]
        std_val = std_20[i]
        upper_bb_val = upper_bb[i]
        lower_bb_val = lower_bb[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        trend_up = close_val > ema_20_4h_aligned[i]
        trend_down = close_val < ema_20_4h_aligned[i]
        
        if position == 0:
            # Long setup: 4h trend up, price at lower BB, RSI oversold, volume confirmation
            if (trend_up and close_val <= lower_bb_val and rsi_val < 30 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: 4h trend down, price at upper BB, RSI overbought, volume confirmation
            elif (trend_down and close_val >= upper_bb_val and rsi_val > 70 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion complete)
            if rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion complete)
            if rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hTrend_BB_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0