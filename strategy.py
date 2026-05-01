#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMMA) turns up AND price > Alligator teeth (8-period SMMA) AND price > 1d EMA50 AND volume > 1.5x 12h volume median.
# Short when Alligator jaws turns down AND price < Alligator teeth AND price < 1d EMA50 AND volume > 1.5x 12h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years) to minimize fee drag.
# Williams Alligator identifies trend inception with smoothed moving averages, reducing false signals.
# 1d EMA50 provides higher-timeframe trend filter, improving robustness in both bull and bear markets.
# Volume confirmation ensures breakouts have genuine participation.

name = "12h_WilliamsAlligator_1dEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams Alligator (SMMA = smoothed moving average)
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full(len(data), np.nan)
        if len(data) < period:
            return sma
        # First value is SMA
        sma[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_DATA) / PERIOD
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Alligator components: Jaws (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume median (30-period for stability)
    vol_median_12h = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Alligator, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 12h volume median
        if vol_median_12h[i] <= 0 or np.isnan(vol_median_12h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_12h[i] * 1.5)
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Alligator signals: Jaws direction and price vs Teeth
        jaw_up = jaw[i] > jaw[i-1]  # Jaws turning up
        jaw_down = jaw[i] < jaw[i-1]  # Jaws turning down
        price_above_teeth = curr_close > teeth[i]
        price_below_teeth = curr_close < teeth[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Jaws turning up AND price above Teeth AND uptrend AND volume confirmation
            if (jaw_up and 
                price_above_teeth and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Jaws turning down AND price below Teeth AND downtrend AND volume confirmation
            elif (jaw_down and 
                  price_below_teeth and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Jaws turning down OR price below Teeth OR trend turns down
            elif (jaw_down or not price_above_teeth or not uptrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Jaws turning up OR price above Teeth OR trend turns up
            elif (jaw_up or not price_below_teeth or not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals