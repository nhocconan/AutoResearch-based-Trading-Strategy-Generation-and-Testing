# NOTE: This strategy is for educational purposes only. Past performance does not guarantee future results.
# Trade responsibly and only with capital you can afford to lose.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Pivot_R1S1_Breakout_Volume_ATRFilter"
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
    
    # Get 1d and 1w data for multi-timeframe analysis (once before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Calculate 1w trend bias (price above/below weekly VWAP approximation)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    # Approximate VWAP: sum(typical_price * volume) / sum(volume)
    # Since we don't have weekly volume in the 1w dataframe, use close as proxy
    vwap_1w = typical_price_1w  # Simplified - in practice would use volume-weighted
    
    # Align 1d levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Align 1w trend bias to 6h timeframe
    vwap_1w_6h = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # 6h ATR for volatility and stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_6h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 2.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or \
           np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(atr_6h[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(vwap_1w_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_6h[i]
        pivot = pivot_6h[i]
        r1 = r1_6h[i]
        s1 = s1_6h[i]
        r4 = r4_6h[i]
        s4 = s4_6h[i]
        weekly_bias = vwap_1w_6h[i]
        
        volume_confirmed = vol > 2.5 * vol_ma
        
        # Determine trend bias from weekly timeframe
        bullish_bias = price > weekly_bias
        bearish_bias = price < weekly_bias
        
        if position == 0:
            # Long: Price breaks above R1 with volume AND weekly bullish bias
            # OR break above R4 (strong breakout) with volume
            if ((price > r1 and volume_confirmed and bullish_bias) or
                (price > r4 and volume_confirmed)):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume AND weekly bearish bias
            # OR break below S4 (strong breakdown) with volume
            elif ((price < s1 and volume_confirmed and bearish_bias) or
                  (price < s4 and volume_confirmed)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot OR ATR stop (2x ATR from entry high)
            # OR weekly bias turns bearish
            if price < pivot or price < (high[i] - 2.0 * atr) or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above pivot OR ATR stop (2x ATR from entry low)
            # OR weekly bias turns bullish
            if price > pivot or price > (low[i] + 2.0 * atr) or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals