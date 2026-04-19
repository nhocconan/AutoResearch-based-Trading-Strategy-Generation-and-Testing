#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and volume confirmation.
# Long when price < 4h VWAP and 1h RSI < 30 and volume > 1.5x average.
# Short when price > 4h VWAP and 1h RSI > 70 and volume > 1.5x average.
# Uses 4h VWAP as dynamic mean, RSI for overextension, volume for confirmation.
# Designed for 1h timeframe to capture mean reversion in ranging markets while
# avoiding counter-trend trades in strong trends. Target: 20-40 trades/year per symbol.
name = "1h_VWAP_RSI_Volume_MeanReversion"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for VWAP trend filter
    df_4h = get_htf_data(prices, '4h')
    typical_price_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3.0
    vwap_4h = (typical_price_4h * df_4h['volume'].values).cumsum() / df_4h['volume'].values.cumsum()
    
    # Align 4h VWAP to 1h
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure RSI and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_4h_aligned[i]
        rsi_val = rsi[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price below VWAP, RSI oversold, and volume confirmation
            if price < vwap and rsi_val < 30 and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Enter short if price above VWAP, RSI overbought, and volume confirmation
            elif price > vwap and rsi_val > 70 and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long when price crosses above VWAP or RSI overbought
            if price > vwap or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when price crosses below VWAP or RSI oversold
            if price < vwap or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals