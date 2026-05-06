# 1d_GoldenCross_Momentum_Exhaustion_v1
# Strategy: 1d Golden Cross + RSI exhaustion with volume confirmation
# - Long when 50-day EMA crosses above 200-day EMA and RSI < 40 (oversold pullback in uptrend)
# - Short when 50-day EMA crosses below 200-day EMA and RSI > 60 (overbought bounce in downtrend)
# - Uses weekly trend filter to avoid counter-trend trades
# - Volume confirmation reduces false signals
# - Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag
# - Works in bull markets (golden cross longs) and bear markets (death cross shorts)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_GoldenCross_Momentum_Exhaustion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for 200 EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly 50 EMA for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily EMAs for golden/death cross
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily RSI for exhaustion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)
    
    # Golden cross: 50 EMA crosses above 200 EMA
    golden_cross = (ema_50 > ema_200) & (np.roll(ema_50, 1) <= np.roll(ema_200, 1))
    golden_cross[0] = False
    
    # Death cross: 50 EMA crosses below 200 EMA
    death_cross = (ema_50 < ema_200) & (np.roll(ema_50, 1) >= np.roll(ema_200, 1))
    death_cross[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Golden cross + RSI oversold (<40) + above weekly EMA50 + volume
            if (golden_cross[i] and rsi[i] < 40 and 
                close[i] > ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Death cross + RSI overbought (>60) + below weekly EMA50 + volume
            elif (death_cross[i] and rsi[i] > 60 and 
                  close[i] < ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought (>70) or death cross
            if rsi[i] > 70 or death_cross[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold (<30) or golden cross
            if rsi[i] < 30 or golden_cross[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals