#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Bollinger Band squeeze and reversal with volume confirmation.
In low volatility (BB width < 20th percentile), look for mean reversion at Bollinger Bands:
- Buy when price touches lower band and closes back inside with volume spike
- Sell when price touches upper band and closes back inside with volume spike
Use 1d EMA200 as trend filter: only take long if price > EMA200, short if price < EMA200.
Exit on opposite band touch or 2x ATR stop.
Designed for 20-50 trades/year to minimize fee flood while capturing mean reversion in ranging markets.
Works in bull markets via buying dips in uptrend and selling rallies in uptrend.
Works in bear markets via selling rallies in downtrend and buying dips in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Bollinger Bands and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band width percentile (20-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan, raw=False
    ).values
    
    # Calculate 1d EMA200 for trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar to close)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_width_pct_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        bb_width_pct_val = bb_width_pct_aligned[i]
        ema_trend = ema_200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price touches lower BB, closes inside, low volatility, uptrend filter, volume spike
            if (prices['low'].iloc[i] <= bb_lower_val and  # touched lower band
                price_close > bb_lower_val and            # closed back inside
                bb_width_pct_val < 0.2 and                # low volatility (BB width < 20th percentile)
                price_close > ema_trend and               # uptrend filter
                vol_ratio_val > 1.5):                     # volume spike
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper BB, closes inside, low volatility, downtrend filter, volume spike
            elif (prices['high'].iloc[i] >= bb_upper_val and  # touched upper band
                  price_close < bb_upper_val and             # closed back inside
                  bb_width_pct_val < 0.2 and                 # low volatility (BB width < 20th percentile)
                  price_close < ema_trend and                # downtrend filter
                  vol_ratio_val > 1.5):                      # volume spike
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite band touch OR ATR-based stoploss
            exit_signal = False
            
            # Opposite band touch exit
            if position == 1 and prices['high'].iloc[i] >= bb_upper_val:
                exit_signal = True
            elif position == -1 and prices['low'].iloc[i] <= bb_lower_val:
                exit_signal =True
            
            # ATR-based stoploss (2x ATR from approximate entry)
            if position == 1:
                # Approximate entry price as the lower BB touch level
                entry_approx = bb_lower_aligned[i-1] if i > 0 else bb_lower_aligned[i]
                if price_close < entry_approx - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # Approximate entry price as the upper BB touch level
                entry_approx = bb_upper_aligned[i-1] if i > 0 else bb_upper_aligned[i]
                if price_close > entry_approx + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_BollingerSqueeze_Reversion_1dEMA200_Volume_ATR"
timeframe = "4h"
leverage = 1.0