#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Donchian channel (20) breakout with volume confirmation and ATR stop.
In uptrend (price > 12h EMA50), buy breakouts above 12h Donchian upper band; in downtrend (price < 12h EMA50), 
sell breakdowns below 12h Donchian lower band. Uses volume to confirm breakout strength and ATR for risk management.
Targets ~20-30 trades/year (80-120 total over 4 years) to avoid fee drag.
Works in bull markets via trend-following breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian channel (20-period)
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (wait for 12h bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # 12h EMA50 for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation (volume spike > 1.8x 30-period average)
    vol_ma_30 = pd.Series(prices['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio = prices['volume'].values / vol_ma_30
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian upper + uptrend + volume spike
            if (price_close > donchian_upper_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian lower + downtrend + volume spike
            elif (price_close < donchian_lower_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR ATR-based stoploss
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry)
            # Note: We approximate entry price as the close that triggered the signal
            # For long: stop if price drops 2*ATR below entry
            # For short: stop if price rises 2*ATR above entry
            if position == 1:
                # Approximate entry price as the Donchian upper breakout level
                entry_approx = donchian_upper_aligned[i-1] if i > 0 else donchian_upper_aligned[i]
                if price_close < entry_approx - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Approximate entry price as the Donchian lower breakdown level
                entry_approx = donchian_lower_aligned[i-1] if i > 0 else donchian_lower_aligned[i]
                if price_close > entry_approx + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0