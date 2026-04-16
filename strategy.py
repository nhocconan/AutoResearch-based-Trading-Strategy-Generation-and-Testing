# 4h_ParabolicSAR_Trend_Scalper_v1
# Hypothesis: Parabolic SAR captures trend direction with built-in acceleration, 
# providing clear entry/exit signals. Combined with volume confirmation and 
# daily trend filter (EMA50), it works in both bull (follows uptrends) and 
# bear (captures downtrends) markets. Using 4h timeframe limits overtrading 
# while capturing multi-day moves. Target: 20-40 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 4h Parabolic SAR (0.02 step, 0.2 max) ===
    # Initialize
    psar = np.zeros(n)
    bull = True  # Start assuming bullish
    af = 0.02    # Acceleration factor
    ep = low[0]  # Extreme point
    psar[0] = ep
    
    for i in range(1, n):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Reverse if price breaks below SAR
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = 0.02
            else:
                # Update EP and AF
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, 0.2)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Reverse if price breaks above SAR
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = 0.02
            else:
                # Update EP and AF
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, 0.2)
    
    # === 1d EMA50 (trend filter) ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h volume confirmation (20-period avg) ===
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume / vol_ma_20_4h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(psar[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        sar = psar[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ratio = vol_ratio_4h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below SAR (trend reversal)
            if price < sar:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above SAR (trend reversal)
            if price > sar:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above SAR AND above daily EMA50 (uptrend) with volume
            if price > sar and price > ema_trend and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Price below SAR AND below daily EMA50 (downtrend) with volume
            elif price < sar and price < ema_trend and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_ParabolicSAR_Trend_Scalper_v1"
timeframe = "4h"
leverage = 1.0