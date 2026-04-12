# 1h_4d_vwap_rsi_divergence_v1
# Hypothesis: 1-hour strategy using VWAP and RSI divergence with daily volume confirmation to catch mean reversions in both bull and bear markets.
# Uses daily VWAP as dynamic support/resistance and RSI(14) divergence for entry timing.
# Volume confirmation from daily timeframe filters low-quality signals.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.

name = "1h_4d_vwap_rsi_divergence_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily VWAP calculation (typical price * volume) / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    
    # Daily RSI(14) for divergence detection
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily VWAP and RSI to 1h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm_1d = volume_1d > (vol_ma_1d * 1.5)
    vol_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d.astype(float))
    
    # 1-hour RSI for entry timing
    delta_h = np.diff(close, prepend=close[0])
    gain_h = np.where(delta_h > 0, delta_h, 0)
    loss_h = np.where(delta_h < 0, -delta_h, 0)
    avg_gain_h = pd.Series(gain_h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_h = pd.Series(loss_h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_h = avg_gain_h / (avg_loss_h + 1e-10)
    rsi_h = 100 - (100 / (1 + rs_h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_confirm_1d_aligned[i]) or np.isnan(rsi_h[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price below VWAP (oversold) AND daily RSI < 30 (oversold) 
        # AND 1h RSI showing bullish divergence (RSI rising while price falling)
        if (close[i] < vwap_1d_aligned[i] and 
            rsi_1d_aligned[i] < 30 and 
            rsi_h[i] > 30 and 
            vol_confirm_1d_aligned[i] > 0.5 and 
            position != 1):
            
            # Check for bullish RSI divergence: current RSI higher than previous low
            lookback = min(10, i)
            if i >= lookback:
                price_low_idx = np.argmin(low[i-lookback:i+1])
                rsi_at_low = rsi_h[i-lookback+price_low_idx]
                if rsi_h[i] > rsi_at_low + 5:  # Bullish divergence
                    position = 1
                    signals[i] = 0.20
        
        # Short condition: price above VWAP (overbought) AND daily RSI > 70 (overbought)
        # AND 1h RSI showing bearish divergence (RSI falling while price rising)
        elif (close[i] > vwap_1d_aligned[i] and 
              rsi_1d_aligned[i] > 70 and 
              rsi_h[i] < 70 and 
              vol_confirm_1d_aligned[i] > 0.5 and 
              position != -1):
            
            # Check for bearish RSI divergence: current RSI lower than previous high
            lookback = min(10, i)
            if i >= lookback:
                price_high_idx = np.argmax(high[i-lookback:i+1])
                rsi_at_high = rsi_h[i-lookback+price_high_idx]
                if rsi_h[i] < rsi_at_high - 5:  # Bearish divergence
                    position = -1
                    signals[i] = -0.20
        
        # Exit conditions
        elif position == 1 and close[i] > vwap_1d_aligned[i]:
            # Exit long when price crosses above VWAP
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < vwap_1d_aligned[i]:
            # Exit short when price crosses below VWAP
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals