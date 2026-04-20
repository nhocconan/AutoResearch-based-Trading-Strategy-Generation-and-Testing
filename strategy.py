#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Momentum with 4h Trend Filter and Volume Confirmation
# - Use 4h EMA (21) as trend filter: long when price > EMA, short when price < EMA
# - Enter on 1h momentum bursts: RSI(14) crossing above 55 (long) or below 45 (short)
# - Require volume > 1.5x 20-period average for confirmation
# - Only trade during active hours (08-20 UTC) to avoid low-liquidity periods
# - Fixed position size of 0.20 to manage risk
# - Designed for 1h timeframe with selective entries to avoid overtrading
# - Target: 15-37 trades per year per symbol (60-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(21) for trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Prepare 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # Determine active session (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if not in active session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if NaN in indicators
        if np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 4h EMA
        price_above_ema = close[i] > ema_21_4h_aligned[i]
        price_below_ema = close[i] < ema_21_4h_aligned[i]
        
        # Momentum signals
        rsi_cross_up = rsi[i] > 55 and rsi[i-1] <= 55
        rsi_cross_down = rsi[i] < 45 and rsi[i-1] >= 45
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long entry: price above 4h EMA + RSI cross up + volume
            if price_above_ema and rsi_cross_up and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short entry: price below 4h EMA + RSI cross down + volume
            elif price_below_ema and rsi_cross_down and vol_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price below 4h EMA or RSI crosses below 50
            if price_below_ema or (rsi[i] < 50 and rsi[i-1] >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price above 4h EMA or RSI crosses above 50
            if price_above_ema or (rsi[i] > 50 and rsi[i-1] <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Momentum_4hEMA_VolumeFilter"
timeframe = "1h"
leverage = 1.0