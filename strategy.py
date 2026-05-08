#3/4/25
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly volume-weighted price action (VWAP) breakout with 1d RSI filter.
# Uses weekly VWAP as dynamic support/resistance, daily volume surge for momentum confirmation,
# and RSI to avoid overextended entries. Designed for low trade frequency (<25/year) to avoid fee drag.
# Works in bull/bear markets by following higher timeframe value areas with strict entry filters.

name = "1d_WeeklyVWAP_Breakout_RSI"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly VWAP (typical price * volume)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_numerator = typical_price_1w * volume_1w
    vwap_denominator = volume_1w
    
    # Calculate cumulative VWAP using expanding window
    vwap_values = np.full_like(close_1w, np.nan)
    cum_numerator = 0.0
    cum_denominator = 0.0
    for i in range(len(close_1w)):
        cum_numerator += vwap_numerator[i]
        cum_denominator += vwap_denominator[i]
        if cum_denominator > 0:
            vwap_values[i] = cum_numerator / cum_denominator
    
    # Align weekly VWAP to daily timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    
    # Get daily data for volume and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Volume spike: 1.5x 20-day EMA (more sensitive than 2x for sufficient signals)
    vol_ema = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ema * 1.5)
    
    # RSI (14-period) with proper Wilder's smoothing
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing: seed with simple average, then smoothed
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        if len(gain) > period:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
            
            for i in range(period + 1, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close_1d, 14)
    
    # Align volume spike and RSI to daily timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above weekly VWAP + volume surge + RSI not overbought
            if close[i] > vwap_aligned[i] and vol_spike_aligned[i] and rsi_aligned[i] < 70:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below weekly VWAP + volume surge + RSI not oversold
            elif close[i] < vwap_aligned[i] and vol_spike_aligned[i] and rsi_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly VWAP or RSI overbought
            if close[i] < vwap_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly VWAP or RSI oversold
            if close[i] > vwap_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals