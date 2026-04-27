#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume spike
# RSI < 30 (oversold) + 4h uptrend + volume spike = long entry
# RSI > 70 (overbought) + 4h downtrend + volume spike = short entry
# Uses 4h EMA for trend, 1h RSI for entry timing, volume filter to avoid false signals
# Timeframe: 1h, targets 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# Works in bull/bear by filtering RSI reversals with 4h trend direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA trend filter (21-period)
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need RSI (14), EMA (21), volume MA (20)
    start_idx = max(14, 21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 4h EMA
        bullish_trend = price > ema_21_4h_aligned[i]
        bearish_trend = price < ema_21_4h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + volume + bullish 4h trend
            if rsi[i] < 30 and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70) + volume + bearish 4h trend
            elif rsi[i] > 70 and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or trend turns bearish
            if rsi[i] >= 50 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or trend turns bullish
            if rsi[i] <= 50 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0