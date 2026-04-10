#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and ATR filter
# - Long when price breaks above Camarilla H4 level with volume > 1.3x 20-day average AND weekly close > weekly open (bullish weekly candle)
# - Short when price breaks below Camarilla L4 level with volume > 1.3x 20-day average AND weekly close < weekly open (bearish weekly candle)
# - Exit when price retests Camarilla H3/L3 levels or ATR(14) expands > 1.5x ATR(50) (volatility expansion signal)
# - Camarilla levels provide intraday support/resistance that work well on daily timeframe
# - Weekly candle direction ensures alignment with higher timeframe momentum
# - Volume confirmation prevents false breakouts
# - ATR filter exits during volatility spikes that often precede reversals
# - Targets 20-30 trades/year (80-120 total over 4 years) to balance opportunity and fee drag

name = "1d_1w_camarilla_breakout_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Pre-compute Camarilla pivot levels from previous day's OHLC
    # Camarilla formula: based on previous day's range
    prev_close = prices['close'].shift(1).values
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h4 = prev_close + (prev_range * 1.1 / 2)  # H4 = C + (R * 1.1/2)
    camarilla_l4 = prev_close - (prev_range * 1.1 / 2)  # L4 = C - (R * 1.1/2)
    camarilla_h3 = prev_close + (prev_range * 1.1 / 4)  # H3 = C + (R * 1.1/4)
    camarilla_l3 = prev_close - (prev_range * 1.1 / 4)  # L3 = C - (R * 1.1/4)
    
    # Pre-compute 1w candle direction (bullish/bearish weekly candle)
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Pre-compute volume confirmation: > 1.3x 20-day average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute ATR filters
    # ATR(14) for current volatility
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # ATR(50) for longer-term volatility average
    atr_50 = pd.Series(true_range).rolling(window=50, min_periods=50).mean().values
    # Volatility expansion: ATR(14) > 1.5 * ATR(50)
    vol_expansion = atr_14 > (1.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(volume_20_avg[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Camarilla H4 with volume spike AND bullish weekly candle
            if (prices['high'].iloc[i] > camarilla_h4[i] and 
                vol_spike.iloc[i] and 
                weekly_bullish_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < Camarilla L4 with volume spike AND bearish weekly candle
            elif (prices['low'].iloc[i] < camarilla_l4[i] and 
                  vol_spike.iloc[i] and 
                  weekly_bearish_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retests Camarilla H3/L3 levels (profit taking/reversal signal)
            # 2. Volatility expansion (ATR(14) > 1.5 * ATR(50)) - exit during volatility spikes
            if position == 1:  # Long position
                if (prices['low'].iloc[i] <= camarilla_h3[i] or 
                    vol_expansion.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] >= camarilla_l3[i] or 
                    vol_expansion.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals