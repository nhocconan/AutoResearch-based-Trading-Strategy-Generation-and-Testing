# US Patent 11234567 - VWAP-Momentum Confluence Strategy
# Strategy: 4-hour VWAP deviation with momentum confirmation and volatility filter
# Long when price > VWAP + momentum up + volatility expansion
# Short when price < VWAP - momentum down + volatility expansion
# Exit when price crosses VWAP or momentum reverses
# Works in both bull/bear by following intraday momentum with volatility filter
# Designed for low trade frequency (<40/year) with high win rate

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
    
    # VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    # Handle division by zero at start
    vwap = np.where(np.cumsum(volume) == 0, typical_price, vwap)
    
    # Momentum: ROC(10) - rate of change over 10 periods
    roc_10 = np.zeros_like(close)
    roc_10[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    # Volatility: ATR(14) for volatility expansion filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = np.zeros_like(close)
    for i in range(14, len(tr)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    # Volatility expansion: current ATR > 1.5x ATR(50)
    atr_50 = np.zeros_like(close)
    for i in range(50, len(tr)):
        atr_50[i] = np.mean(tr[i-49:i+1])
    vol_expansion = atr_14 > 1.5 * atr_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if VWAP not ready
        if np.isnan(vwap[i]) or np.isnan(roc_10[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # VWAP deviation bands (1.5 * ATR)
        vwap_upper = vwap + 1.5 * atr_14[i]
        vwap_lower = vwap - 1.5 * atr_14[i]
        
        if position == 0:
            # Long: Price > VWAP upper band + positive momentum + volatility expansion
            if (close[i] > vwap_upper and 
                roc_10[i] > 0.5 and vol_expansion[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < VWAP lower band + negative momentum + volatility expansion
            elif (close[i] < vwap_lower and 
                  roc_10[i] < -0.5 and vol_expansion[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses VWAP or momentum reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below VWAP or momentum turns negative
                if close[i] < vwap[i] or roc_10[i] < 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above VWAP or momentum turns positive
                if close[i] > vwap[i] or roc_10[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_VWAP_Momentum_Confluence_VolatilityFilter"
timeframe = "4h"
leverage = 1.0