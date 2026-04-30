#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume spike confirmation.
# Long when price is above Alligator lips (SMMA3) with 1w EMA50 uptrend and volume > 1.8x 20-bar average.
# Short when price is below Alligator lips (SMMA3) with 1w EMA50 downtrend and volume spike.
# Uses ATR trailing stop (2.0x) for risk management.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25).
# Williams Alligator identifies trend absence/presence; strong in both bull and bear markets when combined with HTF trend and volume confirmation.
# 1w EMA50 filter ensures alignment with higher-timeframe trend, reducing false signals in ranging markets.

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: SMMA(3) of median price, shifted
    median_price = (high + low) / 2
    smma3 = pd.Series(median_price).ewm(alpha=1/3, adjust=False).mean().values  # SMMA(3) ≈ EMA with alpha=1/3
    smma3_5 = pd.Series(smma3).shift(5).values  # lips: SMMA3 shifted 5 bars
    smma3_8 = pd.Series(smma3).shift(8).values  # teeth: SMMA3 shifted 8 bars
    smma3_13 = pd.Series(smma3).shift(13).values  # jaw: SMMA3 shifted 13 bars
    
    # Volume confirmation: volume > 1.8x 20-period average (balanced to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
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
    
    start_idx = max(50, 20, 13)  # warmup for EMA50, Alligator, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(smma3_5[i]) or np.isnan(smma3_8[i]) or np.isnan(smma3_13[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        # Regime filter: Alligator alignment indicates trend strength
        # Lips above teeth > jaw = uptrend; lips below teeth < jaw = downtrend
        is_uptrend = smma3_5[i] > smma3_8[i] and smma3_8[i] > smma3_13[i]
        is_downtrend = smma3_5[i] < smma3_8[i] and smma3_8[i] < smma3_13[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price above Alligator lips + uptrend + 1w EMA50 uptrend + volume confirmation
            if curr_close > smma3_5[i] and is_uptrend and close[i] > ema_50_1w_aligned[i] and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            # Short: price below Alligator lips + downtrend + 1w EMA50 downtrend + volume confirmation
            elif curr_close < smma3_5[i] and is_downtrend and close[i] < ema_50_1w_aligned[i] and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
        
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