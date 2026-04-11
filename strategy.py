#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + 1d RSI extremes + 1w volume regime filter
# - Enter long when 12h KAMA turns up AND 1d RSI < 30 (oversold) AND 1d volume > 1.5x 20-period volume SMA
# - Enter short when 12h KAMA turns down AND 1d RSI > 70 (overbought) AND 1d volume > 1.5x 20-period volume SMA
# - Exit: opposite KAMA turn or RSI returns to neutral (40-60 range)
# - KAMA adapts to market noise, reducing whipsaws in choppy markets
# - RSI extremes with volume confirmation capture exhaustion moves
# - Volume filter ensures institutional participation
# - Target: 15-35 trades/year to minimize fee drag while capturing high-probability reversals

name = "12h_1d_1w_kama_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for RSI and volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for volume regime filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute KAMA for 12h data (ER=10, fast=2, slow=30)
    # Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of absolute changes
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constant = [ER * (fastest - slowest) + slowest]^2
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA initialization
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at bar 9
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (already in primary TF, no alignment needed)
    # But we need to shift by 1 to avoid look-ahead (KAMA uses current bar's close)
    kama_lagged = np.concatenate([np.full(1, np.nan), kama[:-1]])
    
    # Pre-compute RSI for 1d data (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi_1d = np.concatenate([np.full(14, np.nan), rsi_1d])
    
    # Align RSI to 12h timeframe (wait for completed 1d bar)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d volume aligned for volume confirmation
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Pre-compute volume regime filter for 1w data (volume > 20-period SMA indicates active market)
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_sma_20_1w)
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    volume_regime = volume_1w_aligned > volume_sma_20_1w_aligned  # Active market when volume above average
    
    for i in range(20, n):  # Start after 20-bar warmup for volume SMA
        # Skip if any required data is invalid
        if (np.isnan(kama_lagged[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # KAMA trend: upward when current > previous, downward when current < previous
        kama_up = kama_lagged[i] > kama_lagged[i-1] if i > 0 and not np.isnan(kama_lagged[i-1]) else False
        kama_down = kama_lagged[i] < kama_lagged[i-1] if i > 0 and not np.isnan(kama_lagged[i-1]) else False
        
        # RSI extremes: oversold < 30, overbought > 70
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        rsi_neutral = (rsi_1d_aligned[i] >= 40) & (rsi_1d_aligned[i] <= 60)
        
        # Volume regime: only trade in active markets (1w volume above average)
        active_market = volume_regime[i]
        
        # Trading logic
        if vol_confirm and active_market:
            # Long: KAMA turning up AND RSI oversold
            if kama_up and rsi_oversold:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: KAMA turning down AND RSI overbought
            elif kama_down and rsi_overbought:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits: opposite KAMA turn or RSI returns to neutral
                exit_long = kama_down or rsi_neutral
                exit_short = kama_up or rsi_neutral
                
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation or inactive market: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals