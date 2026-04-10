#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long: price breaks above Camarilla H3 + 4h HMA(21) uptrend + volume > 1.5x 20-period average + session 08-20 UTC
# - Short: price breaks below Camarilla L3 + 4h HMA(21) downtrend + volume > 1.5x 20-period average + session 08-20 UTC
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Stoploss via signal=0 when price breaks opposite Camarilla level (L3 for long, H3 for short)
# - Designed for 1h timeframe: targets 15-37 trades/year to avoid fee drag
# - Camarilla pivots work well in ranging markets; trend filter avoids counter-trend trades
# - Volume confirmation ensures breakout legitimacy
# - Session filter reduces noise during low-liquidity hours

name = "1h_4h_camarilla_breakout_hma_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h HMA(21) for trend filter
    close_4h = df_4h['close'].values
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = np.array([wma(close_4h[i:i+half_len], half_len) if i+half_len <= len(close_4h) else np.nan 
                         for i in range(len(close_4h))])
    wma_full = np.array([wma(close_4h[i:i+21], 21) if i+21 <= len(close_4h) else np.nan 
                         for i in range(len(close_4h))])
    raw_hma = 2 * wma_half - wma_full
    hma_21 = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                       for i in range(len(raw_hma))])
    # Pad beginning with NaN
    hma_21_padded = np.full(len(close_4h), np.nan)
    hma_21_padded[half_len + sqrt_len - 1:len(hma_21)+half_len + sqrt_len - 1] = hma_21
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21_padded)
    
    # Pre-compute 1h Camarilla pivots (based on previous day's high, low, close)
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    # Calculate daily pivot points from previous day's OHLC
    # We need to group by day to get previous day's HLC
    df_1h = prices.copy()
    df_1h['date'] = df_1h['open_time'].dt.date
    prev_day = df_1h.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)  # Previous day's values
    
    # Map previous day's HLC back to 1h bars
    prev_high = df_1h['date'].map(prev_day['high']).values
    prev_low = df_1h['date'].map(prev_day['low']).values
    prev_close = df_1h['date'].map(prev_day['close']).values
    
    # Camarilla pivot calculations
    camarilla_high = np.where(
        ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close),
        prev_close + (prev_high - prev_low) * 1.1 / 6,
        np.nan
    )  # H3 level
    camarilla_low = np.where(
        ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close),
        prev_close - (prev_high - prev_low) * 1.1 / 6,
        np.nan
    )  # L3 level
    
    # Pre-compute 1h volume confirmation
    volume_1h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_spike[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Camarilla L3
            if low_1h[i] < camarilla_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla H3
            if high_1h[i] > camarilla_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above Camarilla H3 + 4h HMA uptrend (close > HMA)
                if high_1h[i] > camarilla_high[i] and close_1h[i] > hma_21_aligned[i]:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: price breaks below Camarilla L3 + 4h HMA downtrend (close < HMA)
                elif low_1h[i] < camarilla_low[i] and close_1h[i] < hma_21_aligned[i]:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals