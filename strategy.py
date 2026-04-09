#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA trend filter and volume confirmation
# Uses Camarilla levels from 1h data for precise entry/exit: L3 breakout = long, H3 breakout = short
# 4h EMA200 filter ensures trades align with higher timeframe trend (avoids counter-trend trades)
# Volume confirmation reduces false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
# Works in bull/bear: EMA200 adapts to trend, Camarilla provides robust support/resistance structure

name = "1h_4h_camarilla_ema_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # for Camarilla calculation
    
    # Load 4h data ONCE before loop for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA200 trend filter
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1h = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup for EMA200
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if EMA data is invalid
        if np.isnan(ema_200_1h[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous bar's OHLC
        if i == 0:
            signals[i] = 0.0
            continue
            
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_open = open_price[i-1]
        
        # Camarilla levels for intraday trading
        # H3 = close + 1.1*(high-low)/2
        # L3 = close - 1.1*(high-low)/2
        # H4 = close + 1.1*(high-low)
        # L4 = close - 1.1*(high-low)
        # Using H3/L3 for entry, H4/L4 for stop (but we use signal=0 for exit)
        range_hl = prev_high - prev_low
        if range_hl <= 0:
            signals[i] = 0.0
            continue
            
        h3 = prev_close + 1.1 * range_hl / 2
        l3 = prev_close - 1.1 * range_hl / 2
        h4 = prev_close + 1.1 * range_hl
        l4 = prev_close - 1.1 * range_hl
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i < 20:
            avg_volume = np.mean(volume[max(0, i-20):i]) if i > 0 else volume[i]
        else:
            avg_volume = np.mean(volume[i-20:i])
        volume_confirm = volume[i] > 1.5 * avg_volume
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR trend turns bearish (price < EMA200)
            if close[i] < l3 or close[i] < ema_200_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR trend turns bullish (price > EMA200)
            if close[i] > h3 or close[i] > ema_200_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with volume and trend confirmation
            if volume_confirm:
                # Long breakout: price closes above H3 AND price > EMA200 (bullish trend)
                if close[i] > h3 and close[i] > ema_200_1h[i]:
                    position = 1
                    signals[i] = 0.20
                # Short breakout: price closes below L3 AND price < EMA200 (bearish trend)
                elif close[i] < l3 and close[i] < ema_200_1h[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals