#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Williams %R measures momentum overbought/oversold levels
# Long when %R crosses above -80 from below AND price > 1d EMA50 AND volume > 2x MA
# Short when %R crosses below -20 from above AND price < 1d EMA50 AND volume > 2x MA
# Uses discrete positions (0.0, ±0.25) to minimize fee churn. Works in both bull/bear by following 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_WilliamsR_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Williams %R (14) on 6h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period_williams = 14
    highest_high = pd.Series(high).rolling(window=period_williams, min_periods=period_williams).max().values
    lowest_low = pd.Series(low).rolling(window=period_williams, min_periods=period_williams).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_williams, 50, 20)  # warmup for Williams %R, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams = williams_r[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Williams %R signals
        williams_oversold = curr_williams < -80
        williams_overbought = curr_williams > -20
        williams_cross_up_oversold = (williams_oversold and i > start_idx and 
                                     williams_r[i-1] >= -80)
        williams_cross_down_overbought = (williams_overbought and i > start_idx and 
                                         williams_r[i-1] <= -20)
        
        # Trend regime: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: Williams %R crosses above -80 from below AND bullish regime
                if williams_cross_up_oversold and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 from above AND bearish regime
                elif williams_cross_down_overbought and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R > -20 (overbought) OR regime changes to bearish
            if curr_williams > -20 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R < -80 (oversold) OR regime changes to bullish
            if curr_williams < -80 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals