#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d/1w Camarilla pivot bounce with volume confirmation and RSI filter.
# Camarilla levels derived from prior day's range provide high-probability reversal points.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation avoids low-conviction bounces.
# RSI filter avoids overbought/oversold extremes.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros(len(close_1w))
    for i in range(20, len(close_1w)):
        ema_1w[i] = np.mean(close_1w[i-20:i])  # Simple average for clarity
    ema_1w = pd.Series(ema_1w[~np.isnan(ema_1w)]).ewm(span=20, adjust=False).mean().values
    ema_1w_full = np.full(len(close_1w), np.nan)
    ema_1w_full[20:] = ema_1w
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_full)
    
    # Calculate RSI(14) for overbought/oversold filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        avg_gain[i] = np.mean(gain[i-14:i])
        avg_loss[i] = np.mean(loss[i-14:i])
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume (20-period) for confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi_val = rsi[i]
        weekly_ema = ema_1w_aligned[i]
        
        # Calculate Camarilla levels from previous day's data
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC from daily data
        prev_day_idx = len(df_1d) - 1
        while prev_day_idx >= 0 and df_1d.iloc[prev_day_idx]['open_time'] >= prices.iloc[i]['open_time']:
            prev_day_idx -= 1
        
        if prev_day_idx < 0:
            signals[i] = 0.0
            continue
            
        prev_high = df_1d.iloc[prev_day_idx]['high']
        prev_low = df_1d.iloc[prev_day_idx]['low']
        prev_close = df_1d.iloc[prev_day_idx]['close']
        
        # Camarilla levels
        range_val = prev_high - prev_low
        h4 = prev_close + range_val * 1.1 / 2
        h3 = prev_close + range_val * 1.1 / 4
        h2 = prev_close + range_val * 1.1 / 6
        h1 = prev_close + range_val * 1.1 / 12
        l1 = prev_close - range_val * 1.1 / 12
        l2 = prev_close - range_val * 1.1 / 6
        l3 = prev_close - range_val * 1.1 / 4
        l4 = prev_close - range_val * 1.1 / 2
        
        # Volume confirmation: current volume > 1.2x average volume
        volume_confirm = vol > 1.2 * avg_vol
        
        if position == 0:
            # Long: price touches L3/L4 with rejection + volume + RSI not oversold + price above weekly EMA
            if (price <= l3 and price > l4 and 
                volume_confirm and 
                rsi_val > 30 and 
                price > weekly_ema):
                position = 1
                signals[i] = position_size
            # Short: price touches H3/H4 with rejection + volume + RSI not overbought + price below weekly EMA
            elif (price >= h3 and price < h4 and 
                  volume_confirm and 
                  rsi_val < 70 and 
                  price < weekly_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 or RSI overbought
            if (price >= h3 or 
                rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 or RSI oversold
            if (price <= l3 or 
                rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Camarilla_Pivot_Bounce_Volume_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0