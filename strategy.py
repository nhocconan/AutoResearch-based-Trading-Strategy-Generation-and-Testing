#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band mean reversion with 1w Supertrend filter and volume spike confirmation.
# Long when price touches lower BB (20,2) and 1w Supertrend is bullish and volume > 1.5x 20-bar average.
# Short when price touches upper BB (20,2) and 1w Supertrend is bearish and volume spike.
# Uses ATR trailing stop (2.0x) for risk management.
# Targets 30-100 total trades over 4 years (7-25/year) with discrete position sizing (0.25).
# Bollinger Bands identify overbought/oversold levels effective in ranging markets, while 1w Supertrend
# filters for higher-timeframe trend alignment to avoid counter-trend trades in strong moves.

name = "1d_BBMeanRev_1wSupertrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Supertrend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend (10,3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_1w = (high_1w + low_1w) / 2
    upper_basic_1w = hl2_1w + 3.0 * atr_1w
    lower_basic_1w = hl2_1w - 3.0 * atr_1w
    
    upper_band_1w = np.zeros_like(upper_basic_1w)
    lower_band_1w = np.zeros_like(lower_basic_1w)
    supertrend_1w = np.zeros_like(close_1w)
    trend_1w = np.ones_like(close_1w, dtype=bool)  # True for uptrend
    
    upper_band_1w[0] = upper_basic_1w[0]
    lower_band_1w[0] = lower_basic_1w[0]
    supertrend_1w[0] = upper_band_1w[0]
    trend_1w[0] = True
    
    for i in range(1, len(close_1w)):
        upper_band_1w[i] = upper_basic_1w[i] if (upper_basic_1w[i] < upper_band_1w[i-1] or close_1w[i-1] > upper_band_1w[i-1]) else upper_band_1w[i-1]
        lower_band_1w[i] = lower_basic_1w[i] if (lower_basic_1w[i] > lower_band_1w[i-1] or close_1w[i-1] < lower_band_1w[i-1]) else lower_band_1w[i-1]
        
        if trend_1w[i-1]:
            supertrend_1w[i] = lower_band_1w[i]
            trend_1w[i] = close_1w[i] >= lower_band_1w[i]
        else:
            supertrend_1w[i] = upper_band_1w[i]
            trend_1w[i] = close_1w[i] <= upper_band_1w[i]
    
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    is_uptrend_1w = supertrend_aligned > close  # price above Supertrend = bullish
    is_downtrend_1w = supertrend_aligned < close  # price below Supertrend = bearish
    
    # Bollinger Bands (20,2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, 20)  # warmup for 1w Supertrend and BB
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(supertrend_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price at lower BB + 1w uptrend + volume confirmation
            if curr_close <= lower_band[i] and is_uptrend_1w[i] and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price at upper BB + 1w downtrend + volume confirmation
            elif curr_close >= upper_band[i] and is_downtrend_1w[i] and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.0 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.0 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals