#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d trend filter + volume spike
# Long when price breaks above Donchian(20) high AND 1d EMA50 uptrend AND volume > 2x MA20
# Short when price breaks below Donchian(20) low AND 1d EMA50 downtrend AND volume > 2x MA20
# Exit when price crosses Donchian midpoint OR trend reverses
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 50-150 total trades over 4 years.

name = "4h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian Channel (20) on 4h data
    period_dc = 20
    highest_high = pd.Series(high).rolling(window=period_dc, min_periods=period_dc).max().values
    lowest_low = pd.Series(low).rolling(window=period_dc, min_periods=period_dc).min().values
    dc_upper = highest_high
    dc_lower = lowest_low
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_dc, 50, 20)  # warmup for Donchian, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_dc_upper = dc_upper[i]
        curr_dc_lower = dc_lower[i]
        curr_dc_middle = dc_middle[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        # Donchian breakout conditions
        breakout_up = curr_close > curr_dc_upper
        breakout_down = curr_close < curr_dc_lower
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: break above upper band AND bullish regime
                if breakout_up and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: break below lower band AND bearish regime
                elif breakout_down and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price < middle OR regime changes to bearish
            if (curr_close < curr_dc_middle) or (not is_bullish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price > middle OR regime changes to bullish
            if (curr_close > curr_dc_middle) or (not is_bearish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals