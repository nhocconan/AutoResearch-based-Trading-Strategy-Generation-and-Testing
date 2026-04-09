#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and ATR stoploss
# - Uses 1d HTF for Camarilla pivot levels (H3/L3) from prior completed daily candle
# - Long when price touches L3 level with volume > 1.5x 20-period average AND close > open (bullish candle)
# - Short when price touches H3 level with volume > 1.5x 20-period average AND close < open (bearish candle)
# - ATR(14) trailing stop: exit long at 2.0x ATR below highest high since entry, exit short at 2.0x ATR above lowest low since entry
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Camarilla levels adapt to volatility, volume/price action filters false touches
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years)

name = "12h_1d_camarilla_volume_priceaction_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3) from prior completed daily candle
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = Pivot + Range * 1.1 / 2
    # L3 = Pivot - Range * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = pivot_1d + range_1d * 1.1 / 2.0
    l3_1d = pivot_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or
            vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Price action confirmation: bullish/bearish candle
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla level touch with volume and price action confirmation
            if volume_confirmed:
                # Long entry: price touches L3 level with bullish candle
                if low[i] <= l3_aligned[i] and bullish_candle:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price touches H3 level with bearish candle
                elif high[i] >= h3_aligned[i] and bearish_candle:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals