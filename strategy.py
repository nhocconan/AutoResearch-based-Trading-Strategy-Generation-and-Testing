#!/usr/bin/env python3
"""
1h_RSI_Trend_With_Volume_Filter_v1
Concept: Use 4h EMA for trend direction, 1h RSI for entry timing, and volume spike for confirmation.
- Long: Price above 4h EMA20 AND RSI(14) crosses above 50 AND volume > 1.5x average
- Short: Price below 4h EMA20 AND RSI(14) crosses below 50 AND volume > 1.5x average
- Exit: RSI crosses back to 50 or opposite extreme (70/30)
- Timeframe: 1h (primary), HTF: 4h for EMA trend
- Position sizing: 0.20 (discrete to minimize fee churn)
- Target: 60-150 trades over 4 years (15-37/year) - avoids fee decay
- Works in bull/bear: EMA filter adapts to trend, RSI captures momentum, volume avoids false signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_RSI_Trend_With_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # === 4h: EMA20 for trend ===
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1h: Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        ema_trend = ema_4h_aligned[i]
        rsi_val = rsi[i]
        rsi_prev = rsi[i-1] if i > 0 else 50
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(rsi_val) or 
            np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        if position == 0:
            # Long: above EMA, RSI crosses above 50, volume confirmation
            if (current_close > ema_trend and 
                rsi_prev < 50 and rsi_val >= 50 and 
                vol_condition):
                signals[i] = 0.20
                position = 1
            # Short: below EMA, RSI crosses below 50, volume confirmation
            elif (current_close < ema_trend and 
                  rsi_prev > 50 and rsi_val <= 50 and 
                  vol_condition):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses below 50 or RSI > 70 (overbought)
            if rsi_val < 50 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI crosses above 50 or RSI < 30 (oversold)
            if rsi_val > 50 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals