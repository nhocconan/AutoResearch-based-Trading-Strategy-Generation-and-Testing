#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w volume regime filter
    # Long: Close breaks above H3 Camarilla level AND 1w volume > 1.5x 20-period average
    # Short: Close breaks below L3 Camarilla level AND 1w volume > 1.5x 20-period average
    # Exit: Close returns to Camarilla pivot level (P) or opposite breakout
    # Using 1d timeframe for low trade frequency, Camarilla for intraday structure,
    # 1w volume for regime filter (avoid low-volume false breakouts), discrete sizing (0.25).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(20, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-20:i])
    volume_regime = vol_1w > (1.5 * vol_ma_1w)
    
    # Align weekly volume regime to 1d
    volume_regime_aligned = align_htf_to_ltf(prices, df_1w, volume_regime)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    camarilla_p = np.full(n, np.nan)   # Pivot
    camarilla_h3 = np.full(n, np.nan)  # High 3
    camarilla_l3 = np.full(n, np.nan)  # Low 3
    camarilla_h4 = np.full(n, np.nan)  # High 4 (stoploss)
    camarilla_l4 = np.full(n, np.nan)  # Low 4 (stoploss)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's Camarilla levels
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        pivot = (phigh + plow + pclose) / 3
        range_ = phigh - plow
        
        camarilla_p[i] = pivot
        camarilla_h3[i] = pivot + (range_ * 1.1 / 4)
        camarilla_l3[i] = pivot - (range_ * 1.1 / 4)
        camarilla_h4[i] = pivot + (range_ * 1.1 / 2)
        camarilla_l4[i] = pivot - (range_ * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_p[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime filter: 1 = high volume (favorable), 0 = low volume (avoid)
        vol_regime = volume_regime_aligned[i] == 1
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h3[i] and vol_regime
        short_breakout = close[i] < camarilla_l3[i] and vol_regime
        
        # Exit conditions: return to pivot or opposite breakout
        long_exit = close[i] < camarilla_p[i] or (close[i] < camarilla_l3[i] and vol_regime)
        short_exit = close[i] > camarilla_p[i] or (close[i] > camarilla_h3[i] and vol_regime)
        
        # Stoploss: H4/L4 breach
        long_stop = close[i] > camarilla_h4[i]
        short_stop = close[i] < camarilla_l4[i]
        
        if (long_breakout or long_stop) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_breakout or short_stop) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0