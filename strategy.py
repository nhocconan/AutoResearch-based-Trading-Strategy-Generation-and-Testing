#!/usr/bin/env python3
"""
Hypothesis: 4h Volume-Weighted Average Price (VWAP) Deviation with 1d ATR Regime Filter
- Uses intraday mean reversion to VWAP during high volatility regimes (1d ATR > 20-period MA)
- Long when price < VWAP - 0.5*ATR(20) in high vol regime, short when price > VWAP + 0.5*ATR(20)
- Volatility regime filter prevents trading in low volatility choppy markets
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in bull markets via mean reversion during pullbacks, in bear markets via bounces in downtrends
"""

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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(20) and its 20-period moving average for regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20 = tr.rolling(window=20, min_periods=20).mean().values
    atr_ma = pd.Series(atr_20).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR regime to 4h timeframe (completed 1d bar only)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Calculate 4h VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 20)  # ATR regime needs 40 bars (20+20), VWAP needs volume history
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_20_aligned[i]) or 
            np.isnan(atr_ma_aligned[i]) or 
            np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # High volatility regime: current ATR > ATR moving average
        high_vol_regime = atr_20_aligned[i] > atr_ma_aligned[i]
        
        # VWAP deviation bands
        upper_band = vwap[i] + 0.5 * atr_20_aligned[i]
        lower_band = vwap[i] - 0.5 * atr_20_aligned[i]
        
        # Mean reversion signals in high volatility regime
        # Long: price below lower band, Short: price above upper band
        long_signal = high_vol_regime and (close[i] < lower_band)
        short_signal = high_vol_regime and (close[i] > upper_band)
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to VWAP or volatility regime ends
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above VWAP or volatility regime ends
                if (close[i] > vwap[i]) or (not high_vol_regime):
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses below VWAP or volatility regime ends
                if (close[i] < vwap[i]) or (not high_vol_regime):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_Deviation_MeanReversion_1dATR_Regime"
timeframe = "4h"
leverage = 1.0