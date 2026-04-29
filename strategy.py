#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakout captures momentum in trending markets, while 1d EMA50 ensures alignment with higher timeframe trend
# Volume confirmation filters false breakouts. Works in both bull and bear markets by following 1d trend direction
# Target: 20-50 trades/year on 4h to minimize fee drag (<10% annual fee cost)
# Stoploss implemented via signal=0 when price breaks opposite Donchian band or regime changes

name = "4h_Donchian20_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above Donchian high AND bullish regime
                if curr_high > curr_donchian_high and is_bullish_regime:
                    signals[i] = 0.30
                    position = 1
                # Bearish entry: price breaks below Donchian low AND bearish regime
                elif curr_low < curr_donchian_low and is_bearish_regime:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below Donchian low OR regime changes to bearish
            if curr_low < curr_donchian_low or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above Donchian high OR regime changes to bullish
            if curr_high > curr_donchian_high or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals