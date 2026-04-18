#!/usr/bin/env python3
"""
1h Intraday Mean Reversion with 1d RSI Filter and Volume Spike
Hypothesis: In 1h timeframe, price often reverts to mean after sharp moves. 
We use 1d RSI to filter regime (RSI<40 for long bias, RSI>60 for short bias) and 
enter on 1h when price deviates >1.5*ATR from VWAP with volume spike (>2x avg volume).
This works in both bull/bear as mean reversion persists across regimes.
Target: 20-40 trades/year to minimize fee drain.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1D INDICATORS (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 14-period RSI on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1H INDICATORS (LTF) ===
    # VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / (np.cumsum(volume) + 1e-10)
    
    # ATR for deviation measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: volume > 2x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2 * vol_ema)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ema[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        atr_val = atr[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        # Deviation from VWAP
        dev_upper = vwap_val + 1.5 * atr_val
        dev_lower = vwap_val - 1.5 * atr_val
        
        if position == 0:
            # Long: RSI<40 (oversold bias) + price below VWAP-1.5*ATR + volume spike
            if rsi_val < 40 and price < dev_lower and vol_ok:
                signals[i] = 0.20
                position = 1
            # Short: RSI>60 (overbought bias) + price above VWAP+1.5*ATR + volume spike
            elif rsi_val > 60 and price > dev_upper and vol_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price returns to VWAP or RSI neutral (40-60)
            if price >= vwap_val or (rsi_val >= 40 and rsi_val <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price returns to VWAP or RSI neutral (40-60)
            if price <= vwap_val or (rsi_val >= 40 and rsi_val <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VWAP_MeanReversion_RSI1D_Volume"
timeframe = "1h"
leverage = 1.0