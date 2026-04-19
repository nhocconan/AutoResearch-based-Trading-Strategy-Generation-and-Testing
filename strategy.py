#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Parabolic SAR + 1w MACD trend filter + volume confirmation
# Parabolic SAR provides trend-following signals with built-in acceleration
# Weekly MACD confirms trend direction to avoid counter-trend trades
# Volume ensures breakout validity. Works in bull/bear by filtering weak signals.
# Target: 15-30 trades/year per symbol.
name = "1d_SAR_MACD1w_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Parabolic SAR calculation
    def calculate_parabolic_sar(high, low, start=0.02, increment=0.02, maximum=0.2):
        n = len(high)
        sar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(n)  # acceleration factor
        ep = np.zeros(n)  # extreme point
        
        # Initialize
        if high[1] > high[0]:
            trend[0] = 1
            sar[0] = low[0]
            ep[0] = high[0]
            af[0] = start
        else:
            trend[0] = -1
            sar[0] = high[0]
            ep[0] = low[0]
            af[0] = start
        
        for i in range(1, n):
            # SAR calculation
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Trend reversal check
            if trend[i-1] == 1:  # uptrend
                if low[i] <= sar[i]:
                    trend[i] = -1  # reverse to downtrend
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = start
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + increment, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # downtrend
                if high[i] >= sar[i]:
                    trend[i] = 1  # reverse to uptrend
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = start
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + increment, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return sar
    
    # Get 1d data for SAR
    sar = calculate_parabolic_sar(high, low)
    
    # Get 1w data for MACD
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate MACD (12,26,9)
    def calculate_macd(close, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        return macd_line, signal_line
    
    macd_line, signal_line = calculate_macd(close_1w)
    
    # Align 1w MACD to 1d
    macd_aligned = align_htf_to_ltf(prices, df_1w, macd_line)
    signal_aligned = align_htf_to_ltf(prices, df_1w, signal_line)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure MACD and SAR are stable
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sar[i]) or np.isnan(macd_aligned[i]) or 
            np.isnan(signal_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sar_val = sar[i]
        macd_val = macd_aligned[i]
        signal_val = signal_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # MACD trend filter: MACD above signal = bullish, below = bearish
        macd_bullish = macd_val > signal_val
        macd_bearish = macd_val < signal_val
        
        if position == 0:
            # Enter long if price above SAR, MACD bullish, and volume confirmation
            if price > sar_val and macd_bullish and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price below SAR, MACD bearish, and volume confirmation
            elif price < sar_val and macd_bearish and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below SAR or MACD turns bearish
            if price < sar_val or not macd_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above SAR or MACD turns bullish
            if price > sar_val or macd_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals