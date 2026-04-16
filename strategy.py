#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume spike confirmation.
# Long when RSI(14) < 30 (oversold) AND 4h EMA50 uptrend (price > EMA50) AND 1h volume > 1.8x 20-period average.
# Short when RSI(14) > 70 (overbought) AND 4h EMA50 downtrend (price < EMA50) AND 1h volume > 1.8x 20-period average.
# Uses discrete position size 0.20. RSI identifies exhaustion points, 4h EMA50 ensures alignment with higher timeframe trend,
# volume spike confirms participation. Session filter (08-20 UTC) reduces noise trades.
# Target: 60-150 trades over 4 years (15-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # === 1h Indicators: RSI (14-period) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1h Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Get 4h data once before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: EMA50 for trend filter ===
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for volume MA, 14 for RSI)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        rsi_val = rsi[i]
        price = close[i]
        ema_4h = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        session_ok = in_session[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI rises above 50 (exiting oversold) or volume spike ends or outside session
            if rsi_val > 50 or not vol_spike or not session_ok:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI falls below 50 (exiting overbought) or volume spike ends or outside session
            if rsi_val < 50 or not vol_spike or not session_ok:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and session_ok:
            # LONG: RSI < 30 (oversold) AND price > 4h EMA50 (uptrend) AND volume spike
            if rsi_val < 30 and price > ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: RSI > 70 (overbought) AND price < 4h EMA50 (downtrend) AND volume spike
            elif rsi_val > 70 and price < ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_VolumeSpike_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0