#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal + 1d EMA50 trend filter + volume spike
- Williams %R(14) identifies overbought/oversold conditions on 12h chart
- Long when %R crosses above -80 from below (oversold bounce) in uptrend (price > 1d EMA50)
- Short when %R crosses below -20 from above (overbought rejection) in downtrend (price < 1d EMA50)
- Volume confirmation (>1.5x 20-period average) ensures institutional participation
- ATR-based stoploss (2.0 * ATR) manages risk
- Target: 12-30 trades/year per symbol (~50-120 total over 4 years)
- Position sizing: 0.25 (discrete levels to minimize fee churn)
- Works in both bull and bear markets: mean reversion in ranges, trend filter avoids whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary calculations (Williams %R, volume, ATR)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 12h
    def calculate_williams_r(high_arr, low_arr, close_arr, window):
        """Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100"""
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        williams_r = (highest_high - close_arr) / (highest_high - lowest_low) * -100
        # Handle division by zero (when high == low)
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
        return williams_r
    
    williams_r = calculate_williams_r(high_12h, low_12h, close_12h, 14)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss on 12h
    def calculate_atr(high_arr, low_arr, close_arr, window):
        """Average True Range"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr
    
    atr_12h = calculate_atr(high_12h, low_12h, close_12h, 14)
    
    # Align all indicators to 12h timeframe (primary timeframe)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Williams %R crossovers
        wr_cross_above_80 = wr > -80 and (i == start_idx or williams_r_aligned[i-1] <= -80)
        wr_cross_below_20 = wr < -20 and (i == start_idx or williams_r_aligned[i-1] >= -20)
        
        if position == 0:
            # Look for reversals with volume confirmation and trend alignment
            # Long: Williams %R crosses above -80 from below + volume spike + price > 1d EMA50 (uptrend)
            if wr_cross_above_80 and vol > 1.5 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -20 from above + volume spike + price < 1d EMA50 (downtrend)
            elif wr_cross_below_20 and vol > 1.5 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Williams %R crosses above -20 (overbought) - take profit
            if wr > -20:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.0 * ATR below entry)
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Williams %R crosses below -80 (oversold) - take profit
            if wr < -80:
                exit_signal = True
            
            # Exit 2: ATR-based stoploss (2.0 * ATR above entry)
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0