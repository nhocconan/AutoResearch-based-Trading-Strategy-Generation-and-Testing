#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above S4 (strong support) AND 1d close > 1d EMA(50) (bullish trend) AND 6h volume > 1.8x 20-bar avg
# - Short when price breaks below R4 (strong resistance) AND 1d close < 1d EMA(50) (bearish trend) AND 6h volume > 1.8x 20-bar avg
# - Exit when price returns to the 1d VWAP (mean reversion to fair value)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla S4/R4 are strong institutional levels where price often reverses or accelerates
# - 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false breakouts
# - VWAP exit provides logical mean reversion target
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, VWAP mean reversion in ranges

name = "6h_1d_camarilla_breakout_vwap_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_bullish = ema_50 > close_1d  # Price above EMA = bullish
    ema_bearish = ema_50 < close_1d  # Price below EMA = bearish
    
    # Align 1d EMA trend to 6h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Pre-compute 1d VWAP for exit
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align 1d VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Pre-compute Camarilla pivot levels from 1d OHLC
    # Camarilla: R4 = close + ((high - low) * 1.1/2), S4 = close - ((high - low) * 1.1/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_range = (high_1d - low_1d) * 1.1 / 2
    r4 = close_1d + camarilla_range
    s4 = close_1d - camarilla_range
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute 6h volume confirmation: > 1.8x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above S4 AND 1d bullish trend AND volume spike
            if (prices['close'].iloc[i] > s4_aligned[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below R4 AND 1d bearish trend AND volume spike
            elif (prices['close'].iloc[i] < r4_aligned[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to VWAP (mean reversion)
            # Exit when price returns to 1d VWAP (within 0.1% tolerance)
            price = prices['close'].iloc[i]
            vwap = vwap_1d_aligned[i]
            exit_signal = np.abs(price - vwap) / vwap < 0.001  # Within 0.1% of VWAP
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals