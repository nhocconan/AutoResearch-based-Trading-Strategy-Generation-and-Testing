#!/usr/bin/env python3
"""
1h_4d_RSI_Divergence_Volume
Hypothesis: On 1h, trade RSI divergences with volume confirmation filtered by 4h trend.
- Bullish divergence: price makes lower low, RSI makes higher low + volume spike
- Bearish divergence: price makes higher high, RSI makes lower high + volume spike
- 4h EMA50 as trend filter: only take long in uptrend, short in downtrend
- Session filter: 08-20 UTC to avoid low-volume Asian session
- Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag
"""

name = "1h_4d_RSI_Divergence_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # --- 4h Trend Filter: EMA50 ---
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # --- RSI(14) on 1h ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume Spike: volume > 1.5 * 20-period average ---
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if session outside 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical values are invalid
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close[i] > ema50_4h_aligned[i]
        trend_down = close[i] < ema50_4h_aligned[i]
        
        if position == 0:
            # Look for bullish divergence: price lower low, RSI higher low
            if i >= 20:  # Need lookback for divergence
                # Find recent swing low in price (last 10 bars)
                lookback = 10
                price_lows = []
                rsi_lows = []
                
                for j in range(i - lookback, i + 1):
                    if j >= 2 and j < len(low) - 2:
                        # Check for pivot low
                        if low[j] <= low[j-1] and low[j] <= low[j-2] and \
                           low[j] <= low[j+1] and low[j] <= low[j+2]:
                            price_lows.append((j, low[j]))
                            rsi_lows.append((j, rsi[j]))
                
                # Need at least 2 lows for divergence
                if len(price_lows) >= 2:
                    last_price_low = price_lows[-1][1]
                    prev_price_low = price_lows[-2][1]
                    last_rsi_low = rsi_lows[-1][1]
                    prev_rsi_low = rsi_lows[-2][1]
                    
                    bullish_div = (last_price_low < prev_price_low and 
                                  last_rsi_low > prev_rsi_low)
                    
                    # Bearish divergence: price higher high, RSI lower high
                    price_highs = []
                    rsi_highs = []
                    
                    for j in range(i - lookback, i + 1):
                        if j >= 2 and j < len(high) - 2:
                            # Check for pivot high
                            if high[j] >= high[j-1] and high[j] >= high[j-2] and \
                               high[j] >= high[j+1] and high[j] >= high[j+2]:
                                price_highs.append((j, high[j]))
                                rsi_highs.append((j, rsi[j]))
                    
                    if len(price_highs) >= 2:
                        last_price_high = price_highs[-1][1]
                        prev_price_high = price_highs[-2][1]
                        last_rsi_high = rsi_highs[-1][1]
                        prev_rsi_high = rsi_highs[-2][1]
                        
                        bearish_div = (last_price_high > prev_price_high and 
                                      last_rsi_high < prev_rsi_high)
                    else:
                        bearish_div = False
                else:
                    bullish_div = False
                    bearish_div = False
            else:
                bullish_div = False
                bearish_div = False
            
            # Enter long on bullish divergence with volume and uptrend
            if bullish_div and vol_spike[i] and trend_up:
                signals[i] = 0.20
                position = 1
            # Enter short on bearish divergence with volume and downtrend
            elif bearish_div and vol_spike[i] and trend_down:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: bearish divergence OR trend breaks down
                if i >= 20:
                    # Quick check for bearish divergence
                    price_highs = []
                    rsi_highs = []
                    lookback = 10
                    for j in range(i - lookback, i + 1):
                        if j >= 2 and j < len(high) - 2:
                            if high[j] >= high[j-1] and high[j] >= high[j-2] and \
                               high[j] >= high[j+1] and high[j] >= high[j+2]:
                                price_highs.append((j, high[j]))
                                rsi_highs.append((j, rsi[j]))
                    
                    bearish_div = False
                    if len(price_highs) >= 2:
                        last_price_high = price_highs[-1][1]
                        prev_price_high = price_highs[-2][1]
                        last_rsi_high = rsi_highs[-1][1]
                        prev_rsi_high = rsi_highs[-2][1]
                        bearish_div = (last_price_high > prev_price_high and 
                                      last_rsi_high < prev_rsi_high)
                else:
                    bearish_div = False
                
                if bearish_div or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: bullish divergence OR trend breaks up
                if i >= 20:
                    # Quick check for bullish divergence
                    price_lows = []
                    rsi_lows = []
                    lookback = 10
                    for j in range(i - lookback, i + 1):
                        if j >= 2 and j < len(low) - 2:
                            if low[j] <= low[j-1] and low[j] <= low[j-2] and \
                               low[j] <= low[j+1] and low[j] <= low[j+2]:
                                price_lows.append((j, low[j]))
                                rsi_lows.append((j, rsi[j]))
                    
                    bullish_div = False
                    if len(price_lows) >= 2:
                        last_price_low = price_lows[-1][1]
                        prev_price_low = price_lows[-2][1]
                        last_rsi_low = rsi_lows[-1][1]
                        prev_rsi_low = rsi_lows[-2][1]
                        bullish_div = (last_price_low < prev_price_low and 
                                      last_rsi_low > prev_rsi_low)
                else:
                    bullish_div = False
                
                if bullish_div or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals