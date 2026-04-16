#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and 1d volume spike.
# Long when RSI(2) < 10 AND price > 4h EMA50 AND 1d volume > 2.0x 20-period average.
# Short when RSI(2) > 90 AND price < 4h EMA50 AND 1d volume > 2.0x 20-period average.
# Exit when RSI(2) crosses above 50 (long) or below 50 (short).
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Position size: 0.20 discrete to minimize fee churn.
# Designed to capture short-term reversals in the direction of the 4h trend with volume confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: EMA50 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # === 1h Indicators: RSI(2) for mean reversion ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(rsi[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        trend_up = price > ema_4h_aligned[i]
        trend_down = price < ema_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI(2) crosses above 50 (mean reversion complete)
            if rsi_val > 50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI(2) crosses below 50 (mean reversion complete)
            if rsi_val < 50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI(2) < 10 AND uptrend AND volume spike
            if rsi_val < 10 and trend_up and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: RSI(2) > 90 AND downtrend AND volume spike
            elif rsi_val > 90 and trend_down and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_RSI2_4hEMA50_1dVolumeSpike_V1"
timeframe = "1h"
leverage = 1.0