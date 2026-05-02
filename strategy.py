#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation + ATR(14) stoploss
# Uses Donchian channels for structure, 12h HMA for trend filter, volume spike for confirmation
# ATR-based stoploss and discrete position sizing (0.25) to minimize fee churn
# Designed for BTC/ETH: works in bull markets via breakouts, bear markets via short breakdowns
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

name = "4h_Donchian20_12hHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h HMA(21)
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    hma_2x = 2 * wma_half - wma_full
    hma_21 = wma(hma_2x, sqrt_len)
    
    # Pad HMA to match original length (due to convolve reducing size)
    hma_21_padded = np.full(len(close_12h), np.nan)
    hma_21_padded[half_len:-sqrt_len+1] = hma_21
    
    # Align 12h HMA to 4h
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_padded)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, ATR, volume MA)
    start_idx = 50  # max(20 for Donchian, 14 for ATR, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h HMA
        trend_up = close[i] > hma_21_aligned[i]
        trend_down = close[i] < hma_21_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper AND trend up AND volume confirm
            if (close[i] > highest_high[i] and 
                trend_up and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND trend down AND volume confirm
            elif (close[i] < lowest_low[i] and 
                  trend_down and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: ATR-based trailing stop
            if close[i] < highest_high[i] - 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: ATR-based trailing stop
            if close[i] > lowest_low[i] + 2.0 * atr_14[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals