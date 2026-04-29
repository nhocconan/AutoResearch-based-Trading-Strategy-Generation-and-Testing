#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper channel in bullish regime (price > weekly EMA50) with volume spike
# Short when price breaks below Donchian lower channel in bearish regime (price < weekly EMA50) with volume spike
# Uses weekly EMA50 to filter for trending markets only, avoiding whipsaws in ranging conditions
# Volume confirmation ensures breakouts have institutional participation
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

name = "1d_Donchian20_1wEMA50_VolumeSpike_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe (completed weekly bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on daily
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for weekly EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_ema_trend = ema_50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: bullish if price > weekly EMA50, bearish if price < weekly EMA50
        is_bullish_trend = curr_close > curr_ema_trend
        is_bearish_trend = curr_close < curr_ema_trend
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian channel in bullish trend
                if is_bullish_trend and curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian channel in bearish trend
                elif is_bearish_trend and curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to middle of channel OR breaks below lower channel with volume
            middle_channel = (curr_upper + curr_lower) / 2.0
            
            if curr_close <= middle_channel or (curr_close < curr_lower and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to middle of channel OR breaks above upper channel with volume
            middle_channel = (curr_upper + curr_lower) / 2.0
            
            if curr_close >= middle_channel or (curr_close > curr_upper and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals