#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume spike
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d EMA34 provides intermediate trend filter to avoid counter-trend trades
# Volume spike confirms institutional participation at reversal points
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Works in both bull and bear markets by following 1d trend while capturing 4h reversals

name = "4h_WilliamsR_1dEMA34_Trend_VolumeSpike_v1"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 4h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume confirmation: volume > 2.0x 20-period average (20*4h = ~3.3 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # warmup for EMA34, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34 = ema34_aligned[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA34, bearish if price < 1d EMA34
        is_bullish_regime = curr_close > curr_ema34
        is_bearish_regime = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: Williams %R oversold (< -80) AND bullish regime
                if curr_williams_r < -80 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R overbought (> -20) AND bearish regime
                elif curr_williams_r > -20 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R becomes overbought (> -20) OR regime changes to bearish
            if curr_williams_r > -20 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R becomes oversold (< -80) OR regime changes to bullish
            if curr_williams_r < -80 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals