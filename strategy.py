#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion + daily EMA34 trend + volume confirmation
# Williams %R identifies overbought/oversold conditions (< -80 oversold, > -20 overbought)
# Daily EMA34 provides trend regime to avoid counter-trend trades
# Volume confirmation ensures mean reversion has follow-through
# Target: 20-30 trades/year (80-120 total over 4 years)

name = "4h_WilliamsR_DailyTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams %R (14-period) on 4h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hl_range = highest_high - lowest_low
    # Avoid division by zero
    williams_r = np.where(hl_range != 0, ((highest_high - close) / hl_range) * -100, -50)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20, 34)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema34_1d = ema34_1d_aligned[i]
        
        # Determine trend regime from daily EMA34
        # Bullish: price > daily EMA34
        # Bearish: price < daily EMA34
        
        if position == 0:  # Flat - look for new entries
            # Williams %R < -80 indicates oversold (long setup)
            # Williams %R > -20 indicates overbought (short setup)
            if curr_wr < -80 and curr_volume_confirm:
                # In bullish regime: look for long setups
                if curr_close > curr_ema34_1d:
                    signals[i] = 0.25
                    position = 1
                # In bearish regime: look for short setups
                elif curr_close < curr_ema34_1d:
                    signals[i] = -0.25
                    position = -1
                # In range/choppy regime (price near EMA): no new entries
            elif curr_wr > -20 and curr_volume_confirm:
                # In bullish regime: look for short setups (fade strength)
                if curr_close < curr_ema34_1d:
                    signals[i] = -0.25
                    position = -1
                # In bearish regime: look for long setups (fade weakness)
                elif curr_close > curr_ema34_1d:
                    signals[i] = 0.25
                    position = 1
                # In range/choppy regime (price near EMA): no new entries
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R > -20 (overbought) OR price crosses below daily EMA34
            if curr_wr > -20 or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R < -80 (oversold) OR price crosses above daily EMA34
            if curr_wr < -80 or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals