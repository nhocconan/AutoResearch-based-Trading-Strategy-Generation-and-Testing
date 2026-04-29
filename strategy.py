#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA50 trend filter and volume confirmation
# Williams %R measures overbought/oversold: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R < -80 (oversold) AND price > 1d EMA50 (bullish trend) AND volume > 1.5x average
# Short when %R > -20 (overbought) AND price < 1d EMA50 (bearish trend) AND volume > 1.5x average
# Exit when %R crosses -50 (mean reversion) or trend changes
# Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year) for optimal fee drag balance
# Works in bull/bear by following 1d trend while capturing 4h mean reversion extremes

name = "4h_WilliamsR_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate Williams %R(14) using 4h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, period)  # warmup for EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: Williams %R < -80 (oversold) AND bullish trend
                if curr_wr < -80 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R > -20 (overbought) AND bearish trend
                elif curr_wr > -20 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R crosses above -50 OR trend changes to bearish
            if curr_wr > -50 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R crosses below -50 OR trend changes to bullish
            if curr_wr < -50 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals