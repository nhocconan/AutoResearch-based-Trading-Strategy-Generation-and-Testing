#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation.
# Long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA) in bull trend (close > 1d EMA34) with volume > 2x 20-period MA.
# Short when jaws cross below teeth in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining sufficient exposure.
# Williams Alligator provides trend-following signals with built-in smoothing to reduce whipsaw.
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "12h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    median_price = (df_12h['high'] + df_12h['low']) / 2
    jaws = smma(median_price.values, 13)  # Blue line
    teeth = smma(median_price.values, 8)   # Red line
    lips = smma(median_price.values, 5)    # Green line (not used)
    
    # Align Alligator lines to 12h timeframe (use prior completed 12h bar)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaws_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        jaw_val = jaws_aligned[i]
        tooth_val = teeth_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Alligator crossover conditions
        bullish_cross = jaw_val > tooth_val
        bearish_cross = jaw_val < tooth_val
        
        # Entry logic
        if position == 0:
            if is_bull_trend and bullish_cross and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and bearish_cross and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish crossover OR trend reversal
            if bearish_cross or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish crossover OR trend reversal
            if bullish_cross or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals