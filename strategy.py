# Hypothetical: 6h_1wVWAP1dRSI_Pullback
# Hypothesis: In BTC/ETH, weekly VWAP acts as institutional support/resistance. Price often pulls back to weekly VWAP before continuing trend.
# Use 1d RSI(14) to confirm momentum alignment during pullback: RSI>50 for longs near VWAP, RSI<50 for shorts.
# Enter on pullback to weekly VWAP with aligned 1d momentum. Exit when price deviates >1.5*ATR from VWAP or RSI diverges.
# Works in bull (buy dips to VWAP) and bear (sell rallies to VWAP). Low frequency: expects 1-2 touches per week per side.

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly VWAP: cumulative (price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pv = (typical_price * df_1w['volume']).cumsum()
    vol_cum = df_1w['volume'].cumsum()
    vwap = pv / vol_cum
    vwap = vwap.replace([np.inf, -np.inf], np.nan).ffill().bfill().values
    
    # Align weekly VWAP to 6s timeframe (no additional delay - VWAP is known within the week)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    
    # Load daily data for RSI and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily RSI(14)
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily ATR(14) for exit condition
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_aligned[i]
        rsi = rsi_1d_aligned[i]
        atr_val = atr[i]
        
        # Distance from VWAP in ATR units
        dist_from_vwap = abs(price - vwap) / atr_val if atr_val > 0 else 0
        
        if position == 0:
            # Long setup: price near weekly VWAP (within 1 ATR) AND bullish momentum (RSI>50)
            if dist_from_vwap <= 1.0 and rsi > 50:
                position = 1
                signals[i] = position_size
            # Short setup: price near weekly VWAP (within 1 ATR) AND bearish momentum (RSI<50)
            elif dist_from_vwap <= 1.0 and rsi < 50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price moves >1.5*ATR away from VWAP OR momentum turns bearish (RSI<40)
            if dist_from_vwap > 1.5 or rsi < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price moves >1.5*ATR away from VWAP OR momentum turns bullish (RSI>60)
            if dist_from_vwap > 1.5 or rsi > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wVWAP1dRSI_Pullback"
timeframe = "6h"
leverage = 1.0