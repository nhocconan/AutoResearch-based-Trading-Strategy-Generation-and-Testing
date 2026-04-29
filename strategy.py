#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and volume spike confirmation
# Long when RSI < 30 AND price > 4h EMA50 AND volume > 2.0x 24-bar avg
# Short when RSI > 70 AND price < 4h EMA50 AND volume > 2.0x 24-bar avg
# Exit when RSI crosses 50 (mean reversion complete) or opposite RSI extreme occurs
# Uses 1h timeframe for precise entry timing, 4h for trend direction
# Session filter: 08-20 UTC to avoid low-liquidity periods
# Discrete position sizing (0.20) minimizes fee drag. Target: 15-35 trades/year on 1h.

name = "1h_RSI14_MeanRev_4hEMA50_VolumeSpike_v1"
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h close prices
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: >2.0x 24-bar average volume (more strict for 1h)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # RSI and volume MA need 24 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi_values[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_ma_24[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_rsi = rsi_values[i]
        ema_50 = ema_50_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when RSI < 30 (oversold) AND price > 4h EMA50 AND volume confirmation
            if curr_rsi < 30 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when RSI > 70 (overbought) AND price < 4h EMA50 AND volume confirmation
            elif curr_rsi > 70 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when RSI crosses 50 (mean reversion) or RSI > 70
            if curr_rsi >= 50 or curr_rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when RSI crosses 50 (mean reversion) or RSI < 30
            if curr_rsi <= 50 or curr_rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals