#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h EMA50 trend filter and 1d volume spike confirmation
# Enter long when: price breaks above upper BB(20,2) AND 4h EMA50 uptrend AND 1d volume > 2x 20-day average
# Enter short when: price breaks below lower BB(20,2) AND 4h EMA50 downtrend AND 1d volume > 2x 20-day average
# Exit when: price returns to middle BB(20) OR trend reverses
# Bollinger squeeze identifies low volatility periods preceding breakouts effective in both bull and bear markets
# Uses 1h for precise entry timing, 4h for trend direction, 1d for volume confirmation to reduce false signals
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag

name = "1h_BB_Squeeze_4hEMA50_Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    middle_bb = sma_20
    
    # Load HTF data ONCE before loop for 4h and 1d calculations
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume confirmation: volume > 2x 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50, 20)  # warmup for BB, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_upper_bb = upper_bb[i]
        curr_lower_bb = lower_bb[i]
        curr_middle_bb = middle_bb[i]
        curr_ema50 = ema50_4h_aligned[i]
        curr_volume_spike = volume_spike_aligned[i] > 0.5  # boolean threshold
        
        # Trend regime: bullish if price > 4h EMA50, bearish if price < 4h EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_spike:
                # Bullish entry: price breaks above upper BB AND bullish regime
                if curr_close > curr_upper_bb and is_bullish_regime:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below lower BB AND bearish regime
                elif curr_close < curr_lower_bb and is_bearish_regime:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to middle BB OR regime changes to bearish
            if curr_close < curr_middle_bb or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to middle BB OR regime changes to bullish
            if curr_close > curr_middle_bb or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals