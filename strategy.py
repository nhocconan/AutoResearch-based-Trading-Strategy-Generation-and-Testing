#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA50 trend filter and volume confirmation.
# Uses daily timeframe for optimal trade frequency (target: 30-100 trades over 4 years).
# Breakouts above 20-day high (long) or below 20-day low (short) with volume spike and weekly trend alignment.
# ATR-based trailing stop for risk management. Discrete sizing 0.25 to balance return and drawdown.
# Weekly HTF ensures we only trade with the higher timeframe trend, reducing whipsaw in ranging markets.

name = "1d_Donchian20_1wHMA50_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w HMA50 trend filter
    close_1w = df_1w['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 50 // 2
    sqrt_len = int(np.sqrt(50))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    wma_half = np.full_like(close_1w, np.nan)
    wma_full = np.full_like(close_1w, np.nan)
    for i in range(half_len, len(close_1w)):
        wma_half[i] = wma(close_1w[i-half_len+1:i+1], half_len)
    for i in range(50, len(close_1w)):
        wma_full[i] = wma(close_1w[i-50+1:i+1], 50)
    
    hma_50 = 2 * wma_half - wma_full
    hma_50 = np.concatenate([np.full(half_len-1, np.nan), hma_50[half_len-1:]])
    hma_50_aligned = align_htf_to_ltf(prices, df_1w, hma_50)
    
    # Calculate daily Donchian channels (20-period)
    # Use prior completed daily bar's high/low for Donchian calculation
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate daily ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):
        # Get current values
        dh = donchian_high[i]
        dl = donchian_low[i]
        ema_trend = hma_50_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(dh) or np.isnan(dl) or np.isnan(ema_trend) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume confirmation: current daily volume > 1.5x 20-period MA
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        volume_spike = volume[i] > (1.5 * vol_ma_20)
        
        # Entry conditions
        # Long: break above 20-day high with volume spike, above weekly HMA50
        long_entry = (close[i] > dh) and volume_spike and (close[i] > ema_trend)
        # Short: break below 20-day low with volume spike, below weekly HMA50
        short_entry = (close[i] < dl) and volume_spike and (close[i] < ema_trend)
        
        # Exit conditions (ATR-based trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals